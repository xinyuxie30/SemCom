"""Post-hoc unit-power-symbol ZF diagnostic, separate from neural feature MSE."""
import json
from pathlib import Path
import numpy as np
from csi_predictors import Predictor, load_trace, samples, predict_at
from run_b0 import write

HERE = Path(__file__).resolve().parent


def main():
    b0 = Path(json.loads((HERE/"latest_b0.json").read_text())["run"])
    if not b0.is_absolute():
        b0 = HERE.parents[1]/b0
    cfg = json.loads((b0/"protocol.json").read_text())
    out = []
    for name in cfg["e2e"]["channels"]:
        path = Path(cfg["baseline_root"])/"SoM-DCAT-main/Channel_inputs"/(name+".mat")
        trace = load_trace(path)
        anchors = samples(trace,cfg["pilot_blocks"]["diagnostic_test"])[0]
        anchors = anchors[np.linspace(0,len(anchors)-1,32).astype(int)]
        model = Predictor.load(b0/name/"wiener_empirical.npz")
        values = {m:[] for m in ("hold","wiener","ideal")}
        amplitudes = {m:[] for m in values}
        for anchor in anchors:
            for phase in cfg["e2e"]["start_phases"]:
                t = anchor*128+phase+np.arange(64)
                truth = trace["CSI"][t]
                estimates = {"hold":trace["NoPred"][t],"wiener":predict_at(trace,model,t,t),"ideal":truth}
                for method,pred in estimates.items():
                    # E|Y/hhat-X|^2 for independent E|X|^2=1 and complex
                    # AWGN variance 10^(-SNR/10). Not actual DCAT latent error.
                    proxy = abs(truth/pred-1)**2 + 10**(-6/10)/abs(pred)**2
                    values[method].extend(proxy.tolist())
                    amplitudes[method].extend(abs(pred).tolist())
        out.append({"channel":name,"snr_db":6,
                    "note":"Expected ZF error for an independent unit-power complex symbol; physics proxy, not measured neural-feature MSE. Same 32 anchors and two phases as E2E.",
                    "methods":{m:{"mean_zf_mse_proxy":float(np.mean(v)),"p95_zf_mse_proxy":float(np.quantile(v,.95)),
                                  "min_estimated_amplitude":float(np.min(amplitudes[m]))} for m,v in values.items()}})
    write(HERE/"zf_diagnostic.json",out)
    print("Saved",HERE/"zf_diagnostic.json")


if __name__=="__main__":
    main()
