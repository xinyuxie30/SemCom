"""Summarize completed B0 runs; exploratory time-block paired intervals."""
import csv
import json
from pathlib import Path
import numpy as np
from csi_predictors import load_trace
from run_b0 import write

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def pointer(filename):
    path = Path(json.loads((HERE/filename).read_text(encoding="utf-8"))["run"])
    return path if path.is_absolute() else REPO/path


def main():
    b0 = pointer("latest_b0.json")
    cfg = json.loads((b0/"protocol.json").read_text(encoding="utf-8"))
    cpu = json.loads((b0/"summary.json").read_text())
    manifests, comparisons, stratified = {}, [], []
    trace_root = Path(cfg["baseline_root"])/"SoM-DCAT-main/Channel_inputs"
    for family, filename in (("without","latest_e2e.json"),("with","latest_e2e_with.json")):
        folder = pointer(filename)
        manifest = json.loads((folder/"manifest.json").read_text())
        assert manifest["status"] == "COMPLETED"
        manifests[family] = {"path":str(folder), **manifest}
        rows = list(csv.DictReader((folder/"per_state.csv").open(encoding="utf-8")))
        assert len(rows) == manifest["expected_rows"] == 6720
        identities = [(r["channel"],r["snr_db"],r["phase"],r["image_id"],r["policy"]) for r in rows]
        assert len(set(identities)) == len(rows)
        numeric = ["mse","psnr_db","lpips_normalized","tx_nmse","rx_nmse","prediction_ms","codec_ms","tx_error_estimate","rx_error_estimate"]
        for row in rows:
            for key in numeric:
                row[key] = float(row[key])
            assert np.isfinite([row[k] for k in numeric]).all()
            assert abs(row["psnr_db"]+10*np.log10(row["mse"])) < 1e-4
        hold = {(r["channel"],r["snr_db"],r["phase"],r["image_id"]):r for r in rows if r["policy"]=="hold"}
        power = {}
        for name in cfg["e2e"]["channels"]:
            trace = load_trace(trace_root/(name+".mat"))
            for r in rows:
                if r["channel"] == name:
                    start = int(r["start"])
                    power[(name,start)] = np.sum(abs(trace["CSI"][start:start+64])**2)
        for name in cfg["e2e"]["channels"]:
            for policy in cfg["e2e"]["policies"]:
                group = [r for r in rows if r["channel"]==name and r["policy"]==policy]
                deltas = [r["psnr_db"]-hold[(name,r["snr_db"],r["phase"],r["image_id"])]["psnr_db"] for r in group]
                scene_deltas = {}
                for r,d in zip(group,deltas):
                    scene_deltas.setdefault(r["scene_id"],{"start":int(r["start"]),"deltas":[]})["deltas"].append(d)
                scene_values = np.array([np.mean(v["deltas"]) for v in sorted(scene_deltas.values(),key=lambda v:v["start"])])
                assert len(scene_values)==32
                # Each block keeps four adjacent sampled channel windows and all
                # repeated SNR/phase observations together. These are NOT new traces.
                block_means = scene_values.reshape(8,4).mean(axis=1)
                rng = np.random.default_rng(20261004)
                bootstrap = block_means[rng.integers(0,8,size=(10000,8))].mean(axis=1)
                ci = np.quantile(bootstrap,[.025,.975]).tolist()
                w = np.array([power[(name,int(r["start"]))] for r in group])
                result = {"checkpoint_family":family,"channel":name,"policy":policy,"states":len(group),
                          "mean_psnr_db":float(np.mean([r["psnr_db"] for r in group])),
                          "delta_psnr_db":float(np.mean(deltas)),"exploratory_block_ci95_db":ci,
                          "mean_mse":float(np.mean([r["mse"] for r in group])),
                          "mean_lpips":float(np.mean([r["lpips_normalized"] for r in group])),
                          "pooled_tx_nmse":float(np.average([r["tx_nmse"] for r in group],weights=w)),
                          "pooled_rx_nmse":float(np.average([r["rx_nmse"] for r in group],weights=w)),
                          "mean_tx_error_estimate":float(np.mean([r["tx_error_estimate"] for r in group])),
                          "mean_rx_error_estimate":float(np.mean([r["rx_error_estimate"] for r in group])),
                          "prediction_ms_median":float(np.median([r["prediction_ms"] for r in group])),
                          "codec_ms_per_image_mean":float(np.mean([r["codec_ms"] for r in group]))}
                comparisons.append(result)
                for snr in cfg["e2e"]["snr_db"]:
                    for phase in cfg["e2e"]["start_phases"]:
                        indexes = [i for i,r in enumerate(group) if float(r["snr_db"])==snr and int(r["phase"])==phase]
                        stratified.append({"checkpoint_family":family,"channel":name,"policy":policy,"snr_db":snr,"phase":phase,
                                           "delta_psnr_db":float(np.mean([deltas[i] for i in indexes]))})
    result = {"status":"VERIFIED_ARITHMETIC_AND_COMPLETED_PILOT_NOT_FORMAL",
              "interval_scope":"Post-hoc exploratory percentile intervals from 10000 resamples of 8 blocks, each comprising 4 adjacent sampled windows/scenes; six SNR/phase repeats stay together. One propagation trace per speed. Not a population CI over new channel realizations; block-size sensitivity untested.",
              "manifests":manifests,"comparisons":comparisons,"stratified":stratified}
    write(HERE/"b0_analysis.json",result)
    with (HERE/"b0_comparisons.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)
    for r in comparisons:
        if r["policy"] in ("hold","wiener_both","acf_both"):
            print(json.dumps({k:r[k] for k in ("checkpoint_family","channel","policy","delta_psnr_db","exploratory_block_ci95_db","pooled_rx_nmse","mean_rx_error_estimate")}))
    make_plot(cpu,comparisons)


def make_plot(cpu,comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    speeds=[2,6,10,15,21]
    names=["A1","A3","A5","A7","A9"]
    c={x["channel"]:x for x in cpu["channels"]}
    d={(r["checkpoint_family"],r["channel"],r["policy"]):r for r in comparisons}
    colors={"wiener_both":"#0072B2","acf_both":"#D55E00","ideal_same_metadata":"#009E73","ideal_zero_age":"#CC79A7"}
    labels={"wiener_both":"Empirical Wiener","acf_both":"ACF Wiener","ideal_same_metadata":"True CSI, same ages","ideal_zero_age":"True CSI, zero ages"}
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(11,7.8),layout="constrained")
    ax=axes[0,0]
    for method,label,color in [("hold","Hold","#555555"),("wiener_empirical","Empirical Wiener","#0072B2"),("wiener_acf","ACF Wiener","#D55E00"),("genie_acf_test_statistics","Oracle test statistics (diagnostic)","#009E73")]:
        ax.semilogy(speeds,[c[n]["diagnostic_test"][method]["rx_nmse"] for n in names],marker="o",label=label,color=color)
    ax.set(title="A. Causal prediction: held-out time block",ylabel="RX NMSE (pooled over all pilot phases)",xlabel="Speed (m/s)")
    ax.legend(fontsize=8)
    for family,ax,title in [("without",axes[0,1],"B. Fixed without_CP checkpoint"),("with",axes[1,0],"C. Fixed with_CP checkpoint (secondary)")]:
        for policy,color in colors.items():
            ax.plot(speeds,[d[family,n,policy]["delta_psnr_db"] for n in names],marker="o",label=labels[policy],color=color)
        ax.axhline(0,color="#777777",linewidth=.8)
        ax.axhline(.3,color="#888888",linestyle=":",linewidth=.8)
        ax.set(title=title,ylabel="Mean PSNR change vs own hold (dB)",xlabel="Speed (m/s)")
        ax.legend(fontsize=8)
    ax=axes[1,1]
    x=np.arange(len(names)); width=.25
    for shift,policy,label,color in [(-1,"wiener_tx","TX only","#56B4E9"),(0,"wiener_rx","RX only","#E69F00"),(1,"wiener_both","TX + RX","#0072B2")]:
        ax.bar(x+shift*width,[d["without",n,policy]["delta_psnr_db"] for n in names],width,label=label,color=color)
    ax.axhline(0,color="#777777",linewidth=.8)
    ax.set(xticks=x,xticklabels=speeds,title="D. Which side changes? without_CP weights",ylabel="Mean PSNR change vs hold (dB)",xlabel="Speed (m/s)")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y",alpha=.18)
    fig.suptitle("B0 feasibility pilot: lower CSI error does not ensure better reconstruction\n32 scenes, 5 released traces, 3 SNRs, 2 start phases; not a formal benchmark",fontsize=12)
    fig.savefig(HERE/"b0_diagnostic.png",dpi=180)
    fig.savefig(HERE/"b0_diagnostic.pdf")
    plt.close(fig)


if __name__=="__main__":
    main()
