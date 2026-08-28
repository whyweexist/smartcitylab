import json, joblib, argparse, pathlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, confusion_matrix, classification_report
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.features import extract_features, FEATURE_NAMES, features_to_vector

ISSUE_TYPES = ["blur","underexposure","overexposure","noise","severe_degradation","potential_defect"]

def evaluate(manifest="data/manifest.json", image_dir="data/generated", model="backend/artifacts/model.joblib", out_dir="data/evaluation"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest) as f:
        man = json.load(f)
    bundle = joblib.load(model)
    scaler = bundle["scaler"]; clf = bundle["clf"]; iso = bundle["iso"]

    # Collect test only
    X_list=[]; y_true=[]; splits=[]
    import cv2
    from app.preprocessing import safe_decode
    for entry in man:
        if entry["split"]!="test":
            continue
        ipath = Path(image_dir)/entry["image_id"]
        if not ipath.exists():
            continue
        data = ipath.read_bytes()
        try:
            rgb,gray,_ = safe_decode(data)
        except:
            continue
        feats = extract_features(rgb,gray)
        vec = features_to_vector(feats)
        X_list.append(vec)
        issues=set(entry["issues"])
        y_vec=[1 if k in issues else 0 for k in ISSUE_TYPES]
        y_true.append(y_vec)
    X = np.stack(X_list) if X_list else np.zeros((0,len(FEATURE_NAMES)))
    y_true = np.array(y_true)
    Xs = scaler.transform(X) if len(X) else X
    y_pred = clf.predict(Xs) if len(X) else y_true
    # also anomaly prob for potential_defect evaluation
    decs = iso.decision_function(Xs) if len(X) else np.array([])
    anomaly = 1/(1+np.exp(decs*3)) if len(decs) else np.array([])

    metrics={}
    per_class=[]
    for i,name in enumerate(ISSUE_TYPES):
        p,r,f,_ = precision_recall_fscore_support(y_true[:,i], y_pred[:,i], zero_division=0, average="binary")
        per_class.append({"issue":name,"precision":p,"recall":r,"f1":f})
        metrics[name]={"precision":p,"recall":r,"f1":f}

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_f1"]=float(macro_f1)
    metrics["micro_f1"]=float(micro_f1)
    print("Per class:")
    for c in per_class: print(c)
    print("macro",macro_f1,"micro",micro_f1)

    # Overall quality label accuracy (derive from manifest vs predicted label via scoring)
    # For evaluation we map quality label to 0/1/2
    label_map={"ACCEPTABLE":0,"DEGRADED":1,"POTENTIALLY_DEFECTIVE":2}
    y_q_true=[]; y_q_pred=[]
    # Need to reconstruct quality scoring for each test sample
    from app.scoring import compute_quality_score, classify_label
    for idx, entry in enumerate([e for e in man if e["split"]=="test"]):
        if idx>=len(y_true): break
        true_label = entry["quality_label"]
        # predicted probs
        probs={name: float(y_pred[idx,i]) for i,name in enumerate(ISSUE_TYPES)}  # using hard 0/1 as proxy, but use clf predict_proba if available
        # Use actual proba for better scoring
        vec = Xs[idx:idx+1]
        prob_dict={}
        for ci, name in enumerate(ISSUE_TYPES):
            est=clf.estimators_[ci]
            if hasattr(est,"predict_proba"):
                proba=est.predict_proba(vec)[0]
                if len(proba)==2:
                    p = float(proba[list(est.classes_).index(1)] if 1 in est.classes_ else 0)
                else:
                    p=float(proba[0])
            else:
                p=float(y_pred[idx,ci])
            prob_dict[name]=p
        anom = float(anomaly[idx]) if len(anomaly)>idx else 0.0
        score=compute_quality_score(prob_dict, anom)
        pred_label=classify_label(score, prob_dict, anom)
        y_q_true.append(label_map.get(true_label,1))
        y_q_pred.append(label_map.get(pred_label,1))

    if y_q_true:
        acc=accuracy_score(y_q_true, y_q_pred)
        macro_f1_q = f1_score(y_q_true, y_q_pred, average="macro", zero_division=0)
        cm=confusion_matrix(y_q_true, y_q_pred, labels=[0,1,2])
        metrics["quality_accuracy"]=float(acc)
        metrics["quality_macro_f1"]=float(macro_f1_q)
        metrics["confusion_matrix"]=cm.tolist()
        print("Quality acc",acc,"macroF1",macro_f1_q)
        print(cm)
        # plot confusion matrix
        fig,ax=plt.subplots(figsize=(5,4))
        im=ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
        ax.set_xticklabels(["ACC","DEG","DEF"]); ax.set_yticklabels(["ACC","DEG","DEF"])
        ax.set_xlabel("Pred"); ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                ax.text(j,i, cm[i,j], ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        # classification report
        rep=classification_report(y_q_true, y_q_pred, target_names=["ACCEPTABLE","DEGRADED","POTENTIALLY_DEFECTIVE"], output_dict=True, zero_division=0)
        metrics["classification_report"]=rep

    # Feature importance plot
    importances=bundle.get("importances",{})
    if importances:
        fig,ax=plt.subplots(figsize=(10,4))
        width=0.12
        x=np.arange(len(FEATURE_NAMES))
        for i,name in enumerate(ISSUE_TYPES):
            vals=np.array(importances[name])
            ax.bar(x+i*width, vals, width, label=name)
        ax.set_xticks(x+width*2.5)
        ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir/"feature_importance.png", dpi=150)
        plt.close()

    # Baseline vs ML comparison (simple threshold baseline using sharpness/brightness)
    # baseline: blur if laplacian_var <80, underexposed if brightness_mean<60, overexposed >200, noise if noise_est>12
    # compute baseline f1 for comparison
    # Need feats again for test set: re-extract brightness etc
    # For speed, approximate using same X indices mapping to feats: we have X raw vectors; map indices to features
    # Feature index map
    fidx={n:i for i,n in enumerate(FEATURE_NAMES)}
    y_base=np.zeros_like(y_true)
    for i in range(len(X)):
        vec=X[i]
        if vec[fidx["laplacian_var"]]<80: y_base[i,0]=1
        if vec[fidx["brightness_mean"]]<70 or vec[fidx["dark_ratio"]]>0.35: y_base[i,1]=1
        if vec[fidx["brightness_mean"]]>190 or vec[fidx["bright_ratio"]]>0.30: y_base[i,2]=1
        if vec[fidx["noise_est"]]>12: y_base[i,3]=1
        if vec[fidx["blockiness"]]>0.6 or vec[fidx["hf_ratio"]]<0.35: y_base[i,4]=1
        # potential defect baseline from iso anomaly >0.5
        if anomaly[i]>0.5: y_base[i,5]=1
    base_macro=f1_score(y_true, y_base, average="macro", zero_division=0)
    base_micro=f1_score(y_true, y_base, average="micro", zero_division=0)
    metrics["baseline_macro_f1"]=float(base_macro)
    metrics["baseline_micro_f1"]=float(base_micro)
    print(f"Baseline macro {base_macro:.3f} vs ML macro {macro_f1:.3f}")

    # Save metrics
    with open(out_dir/"metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    # Human summary
    with open(out_dir/"evaluation_summary.md","w") as f:
        f.write("# Evaluation Summary\n\n")
        f.write(f"Test samples: {len(y_true)}\n\n")
        f.write("## Per-issue F1\n")
        for c in per_class:
            f.write(f"- {c['issue']}: P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f}\n")
        f.write(f"\nMacro F1: {macro_f1:.3f}\n")
        f.write(f"Micro F1: {micro_f1:.3f}\n")
        f.write(f"Baseline Macro F1: {base_macro:.3f}\n")
        if "quality_accuracy" in metrics:
            f.write(f"\nQuality Accuracy: {metrics['quality_accuracy']:.3f}\n")
            f.write(f"Quality Macro F1: {metrics['quality_macro_f1']:.3f}\n")
        f.write("\n## Notes\n")
        f.write("- Leakage-safe: split by source clean id before synthesis.\n")
        f.write("- Anomaly detector trained only on clean training samples.\n")

    # Failure examples: save misclassified indices
    failures=[]
    for i in range(len(y_true)):
        if not np.array_equal(y_true[i], y_pred[i]):
            failures.append({"idx":int(i),"true":y_true[i].tolist(),"pred":y_pred[i].tolist()})
            if len(failures)>=20: break
    with open(out_dir/"failures.json","w") as f:
        json.dump(failures,f,indent=2)

    return metrics

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--image_dir", default="data/generated")
    ap.add_argument("--model", default="backend/artifacts/model.joblib")
    ap.add_argument("--out", default="data/evaluation")
    args=ap.parse_args()
    evaluate(args.manifest, args.image_dir, args.model, args.out)
