from __future__ import annotations

import unittest
from pathlib import Path
import sys
import tempfile

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.slow_loop.dynamic_kb_pruner import DynamicKBPruner, GlobalKnowledgeGraph
from agents.slow_loop.skb_server import CloudSKBServer, EdgeSKBServer
from data.dataloader import STATE_DIM
from data.kg_interface import INTENT_DIM, SubGraph
from env.mdp_env import (
    EnvConfig,
    NODE_EMB_DIM,
    SemiMDPEnv,
    encode_semimdp_action,
)
from models.transe import TransE
from models.transe_embedder import (
    TransEEmbedder,
    _entity2id_fingerprint,
    _resolve_checkpoint_paths,
)
from utils.semantic_consistency import kg_semantic_consistency


class FakeKeywordEmbedder:
    def __init__(self) -> None:
        self.dim = 4
        self.entity_vectors = {
            "alpha": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "beta": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "gamma": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        }
        self.keyword_vectors = {
            "alpha_word": (self.entity_vectors["alpha"], False),
            "mystery": (np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), True),
        }

    def embed_entity(self, entity_id: str) -> np.ndarray:
        return self.entity_vectors[entity_id].copy()

    def embed_keyword(self, keyword: str) -> tuple[np.ndarray, bool]:
        vec, is_ood = self.keyword_vectors[keyword]
        return vec.copy(), is_ood


def build_toy_kg() -> GlobalKnowledgeGraph:
    kg = GlobalKnowledgeGraph("unused")
    kg.entity2id = {"alpha": 0, "beta": 1, "gamma": 2}
    kg.id2entity = {idx: entity for entity, idx in kg.entity2id.items()}
    kg.relation2id = {"rel": 0}
    kg.id2relation = {0: "rel"}
    kg.graph = nx.DiGraph()
    kg.graph.add_edge("alpha", "beta", relation="rel")
    kg.graph.add_edge("beta", "gamma", relation="rel")
    kg.graph.add_edge("gamma", "alpha", relation="rel")
    return kg


class TransEPipelineTests(unittest.TestCase):
    def test_model_scores_use_normalized_embeddings_by_default(self) -> None:
        model = TransE(2, 1, embedding_dim=2)
        with torch.no_grad():
            model.entity_emb.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 4.0]]))
            model.relation_emb.weight.copy_(torch.tensor([[3.0, 0.0]]))

        score = model(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),
        )

        expected = torch.norm(
            torch.tensor([[1.0, 0.0]]) + torch.tensor([[1.0, 0.0]]) - torch.tensor([[0.0, 1.0]]),
            p=1,
            dim=-1,
        )
        self.assertTrue(torch.allclose(score, expected))
        self.assertEqual(model.margin, 5.0)

    def test_transe_embedder_uses_entity_ids_as_embedding_rows(self) -> None:
        entity_emb = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        embedder = TransEEmbedder(entity_emb, {"entity_b": 1, "entity_a": 0, "entity_c": 2})
        np.testing.assert_allclose(embedder.embed_entity("entity_b"), np.array([0.0, 1.0], dtype=np.float32))

    def test_entity_fingerprint_separates_same_size_datasets(self) -> None:
        fp_a = _entity2id_fingerprint({"00001740n": 0, "00002098n": 1})
        fp_b = _entity2id_fingerprint({"alpha": 0, "beta": 1})
        self.assertNotEqual(fp_a, fp_b)

    def test_checkpoint_resolver_requires_suffix_when_multiple_seeds_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp)
            for seed in ("seed42", "seed123"):
                (ckpt / f"transe_fb15k237_transe_{seed}_entity_emb.npy").write_bytes(b"")
                (ckpt / f"transe_fb15k237_transe_{seed}_entity2id.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                _resolve_checkpoint_paths(str(ckpt), "FB15K237", "transE")

            emb_path, e2id_path = _resolve_checkpoint_paths(
                str(ckpt), "FB15K237", "transE", tag_suffix="seed42"
            )
            self.assertEqual(emb_path.name, "transe_fb15k237_transe_seed42_entity_emb.npy")
            self.assertEqual(e2id_path.name, "transe_fb15k237_transe_seed42_entity2id.json")

    def test_pruner_matches_known_keyword_and_directly_injects_ood(self) -> None:
        kg = build_toy_kg()
        pruner = DynamicKBPruner(
            kg,
            max_subgraph_nodes=8,
            similarity_threshold=0.8,
            transe_embedder=FakeKeywordEmbedder(),
        )

        seed_nodes, injected = pruner._match_and_inject(["alpha_word", "mystery"])

        self.assertIn("alpha", seed_nodes)
        self.assertIn("mystery", seed_nodes)
        self.assertEqual(injected, 1)
        self.assertIn("mystery", kg.entity2id)
        mystery_trace = [row for row in pruner._match_trace if row["keyword"] == "mystery"][0]
        self.assertTrue(mystery_trace["is_ood"])
        self.assertEqual(mystery_trace["action"], "inject")
        self.assertIsNone(mystery_trace["best_match"])

    def test_pruner_reuses_cached_transe_matrix(self) -> None:
        kg = build_toy_kg()
        pruner = DynamicKBPruner(kg, transe_embedder=FakeKeywordEmbedder())

        eid_list_1, matrix_1 = pruner._get_transe_entity_matrix()
        kg.inject_node("dynamic_node")
        eid_list_2, matrix_2 = pruner._get_transe_entity_matrix()

        self.assertIs(eid_list_1, eid_list_2)
        self.assertIs(matrix_1, matrix_2)
        self.assertNotIn("dynamic_node", eid_list_2)

    def test_subgraph_exports_semantic_node_embeddings(self) -> None:
        sub_graph = SubGraph(
            nodes=["alpha", "beta"],
            edges=[("alpha", "rel", "beta")],
            intent_vec=np.zeros(INTENT_DIM, dtype=np.float32),
            node_embeddings={
                "alpha": np.ones(4, dtype=np.float32),
                "beta": -np.ones(4, dtype=np.float32),
            },
        )

        tensors = sub_graph.to_tensors(embedding_dim=8)

        self.assertEqual(tuple(tensors["node_embeddings"].shape), (2, 8))
        np.testing.assert_allclose(
            tensors["node_embeddings"][0, :4].numpy(),
            np.ones(4, dtype=np.float32),
        )
        np.testing.assert_allclose(
            tensors["node_embeddings"][0, 4:].numpy(),
            np.zeros(4, dtype=np.float32),
        )

    def test_subgraph_rejects_non_contract_intent_dim(self) -> None:
        with self.assertRaises(ValueError):
            SubGraph(
                nodes=["alpha"],
                edges=[],
                intent_vec=np.zeros(INTENT_DIM + 1, dtype=np.float32),
            )

    def test_env_uses_semantic_node_embeddings_and_resource_action(self) -> None:
        intent = np.zeros(INTENT_DIM, dtype=np.float32)
        sub_graph = SubGraph(
            nodes=["alpha", "beta"],
            edges=[("alpha", "rel", "beta")],
            intent_vec=intent,
            node_embeddings={
                "alpha": np.ones(128, dtype=np.float32),
                "beta": -np.ones(128, dtype=np.float32),
            },
        )
        env = SemiMDPEnv(EnvConfig(snr_db_low=10.0, snr_db_high=10.0))
        state = env.reset(sub_graph, intent)

        self.assertEqual(state["node_emb"].shape[0], NODE_EMB_DIM)
        self.assertEqual(env.state_dim, STATE_DIM)
        action = encode_semimdp_action(
            node_idx=0,
            num_neighbours=1,
            bw_idx=5,
            mcs_idx=2,
            action_dim=env.action_dim,
            num_bw_bins=env.cfg.num_bw_bins,
            num_mcs_levels=env.cfg.num_mcs_levels,
        )
        _, reward, done, info = env.step(action)

        self.assertTrue(done)
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["bw_idx"], 5)
        self.assertEqual(info["mcs_idx"], 2)

    def test_env_reset_uses_subgraph_intent_when_not_overridden(self) -> None:
        intent = np.zeros(INTENT_DIM, dtype=np.float32)
        sub_graph = SubGraph(
            nodes=["alpha", "beta"],
            edges=[("alpha", "rel", "beta")],
            intent_vec=intent,
        )
        env = SemiMDPEnv(EnvConfig())
        state = env.reset(sub_graph)

        self.assertEqual(state["node_emb"].shape[0], NODE_EMB_DIM)
        self.assertEqual(env.state_dim, STATE_DIM)
        self.assertEqual(env.observation_dim, NODE_EMB_DIM + env.cfg.csi_dim)

    def test_incremental_skb_alignment_restores_consistency_with_lower_payload(self) -> None:
        intent = np.zeros(INTENT_DIM, dtype=np.float32)
        reference = SubGraph(
            nodes=["alpha", "beta", "gamma"],
            edges=[("alpha", "rel", "beta"), ("beta", "rel", "gamma")],
            intent_vec=intent,
            metadata={
                "match_trace": [
                    {"action": "match", "best_match": "alpha"},
                    {"action": "inject", "keyword": "gamma"},
                ]
            },
            node_embeddings={
                "alpha": np.ones(4, dtype=np.float32),
                "beta": np.ones(4, dtype=np.float32) * 2,
                "gamma": np.ones(4, dtype=np.float32) * 3,
            },
        )
        cloud = CloudSKBServer(node_id_bytes=6, relation_id_bytes=3, embedding_dim=4)
        stale = EdgeSKBServer(capacity_nodes=8)
        stale.cached_nodes.add("alpha")
        stale.cached_embeddings["alpha"] = reference.node_embeddings["alpha"].copy()

        no_align_edge = stale.clone()
        no_align_report = no_align_edge.align(reference, cloud, "no_align")
        no_align_graph = no_align_edge.materialize(reference)
        no_align_score = kg_semantic_consistency(reference, no_align_graph)["semantic_consistency"]

        incremental_edge = stale.clone()
        incremental_report = incremental_edge.align(reference, cloud, "incremental_align")
        incremental_graph = incremental_edge.materialize(reference)
        incremental_score = kg_semantic_consistency(reference, incremental_graph)["semantic_consistency"]

        full_edge = stale.clone()
        full_report = full_edge.align(reference, cloud, "full_retransmit")

        self.assertLess(no_align_score, 1.0)
        self.assertEqual(incremental_score, 1.0)
        self.assertGreater(incremental_report.payload_bytes, no_align_report.payload_bytes)
        self.assertLess(incremental_report.payload_bytes, full_report.payload_bytes)
        self.assertGreater(incremental_report.kappa, 0.0)


if __name__ == "__main__":
    unittest.main()
