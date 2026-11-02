import os, sys, random, subprocess, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, MACCSkeys, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","rdkit"])
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, MACCSkeys, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

SEED=42; DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SPLITS=5; AUX_SAMPLE=50000
print("device:",DEVICE)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def find_file(name):
    if Path(name).exists(): return Path(name)
    hits=list(Path("/kaggle/input").rglob(name))
    if not hits: raise FileNotFoundError(name)
    primary=[p for p in hits if "archive" not in str(p).lower()]
    return primary[0] if primary else hits[0]

def find_archive_file(name):
    hits=[p for p in Path("/kaggle/input").rglob(name) if "archive" in str(p).lower()]
    if hits: return hits[0]
    local=Path("archive")/name
    return local if local.exists() else None

def safe_mol(smiles):
    mol=Chem.MolFromSmiles(str(smiles))
    return mol if mol is not None else Chem.MolFromSmiles("*")

def canonical_key(smiles):
    try:
        mol=Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True) if mol is not None else ""
    except Exception:
        return ""

def scaffold_group(smiles):
    mol=safe_mol(smiles)
    try:
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaffold: return scaffold
        return Chem.MolToSmiles(mol,canonical=True)
    except Exception:
        return str(smiles)

DESC_NAMES=["MolWt","LogP","TPSA","Heavy","Rings","AromRings","Rotors","HBA","HBD",
            "CSP3","AromFrac","ConjFrac","RingBondFrac","Backbone","BackboneConj",
            "BackboneArom","BackboneHetero","OffBackbone"]

def molecular_features(smiles_values, include_descriptors=True):
    fp=np.zeros((len(smiles_values),2048),np.uint8)
    desc=np.zeros((len(smiles_values),len(DESC_NAMES)),np.float32) if include_descriptors else None
    for i,s in enumerate(smiles_values):
        mol=safe_mol(s); bit=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
        DataStructs.ConvertToNumpyArray(bit,fp[i])
        if not include_descriptors: continue
        atoms=list(mol.GetAtoms()); bonds=list(mol.GetBonds()); na=max(len(atoms),1); nb=max(len(bonds),1)
        arom=sum(a.GetIsAromatic() for a in atoms); conj=sum(b.GetIsConjugated() or b.GetIsAromatic() for b in bonds)
        dummy=[a.GetIdx() for a in atoms if a.GetAtomicNum()==0]; backbone=bc=ba=bh=off=0.0
        if len(dummy)>=2:
            path=list(Chem.GetShortestPath(mol,dummy[0],dummy[1])); ps=set(path); pb=[]
            for a,b in zip(path[:-1],path[1:]):
                bond=mol.GetBondBetweenAtoms(a,b)
                if bond is not None: pb.append(bond)
            internal=[mol.GetAtomWithIdx(j) for j in path if mol.GetAtomWithIdx(j).GetAtomicNum()!=0]
            backbone=len(pb); den=max(len(pb),1); aden=max(len(internal),1)
            bc=sum(b.GetIsConjugated() or b.GetIsAromatic() for b in pb)/den
            ba=sum(b.GetIsAromatic() for b in pb)/den
            bh=sum(a.GetAtomicNum() not in (0,6) for a in internal)/aden
            off=sum(a.GetIdx() not in ps for a in atoms)/na
        values=[Descriptors.MolWt(mol),Crippen.MolLogP(mol),rdMolDescriptors.CalcTPSA(mol),
                mol.GetNumHeavyAtoms(),rdMolDescriptors.CalcNumRings(mol),rdMolDescriptors.CalcNumAromaticRings(mol),
                Lipinski.NumRotatableBonds(mol),Lipinski.NumHAcceptors(mol),Lipinski.NumHDonors(mol),
                Lipinski.FractionCSP3(mol),arom/na,conj/nb,sum(b.IsInRing() for b in bonds)/nb,
                backbone,bc,ba,bh,off]
        desc[i]=np.nan_to_num(values,nan=0,posinf=0,neginf=0)
    return fp,desc

def rich_fingerprints(smiles_values):
    blocks=[np.zeros((len(smiles_values),n),np.uint8) for n in [512,1024,1024,1024,1024,167]]
    for i,s in enumerate(smiles_values):
        mol=safe_mol(s)
        fps=[AllChem.GetMorganFingerprintAsBitVect(mol,1,nBits=512),
             AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024),
             AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=1024),
             Chem.RDKFingerprint(mol,fpSize=1024,maxPath=7),
             rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol,nBits=1024),
             MACCSkeys.GenMACCSKeys(mol)]
        for block,bit in zip(blocks,fps): DataStructs.ConvertToNumpyArray(bit,block[i])
    return np.hstack(blocks).astype(np.float32)

def full_rdkit_descriptor_pair(train_smiles,test_smiles):
    funcs=list(Descriptors._descList); rows=[]
    for s in list(train_smiles)+list(test_smiles):
        mol=safe_mol(s); values=[]
        for _,fn in funcs:
            try:
                value=float(fn(mol)); values.append(value if np.isfinite(value) else np.nan)
            except Exception: values.append(np.nan)
        rows.append(values)
    frame=pd.DataFrame(rows,dtype=np.float32).replace([np.inf,-np.inf],np.nan)
    tr=frame.iloc[:len(train_smiles)].reset_index(drop=True); te=frame.iloc[len(train_smiles):].reset_index(drop=True)
    keep=(tr.notna().mean()>=.80)&(tr.nunique(dropna=True)>1); tr=tr.loc[:,keep]; te=te.loc[:,keep]
    med=tr.median().fillna(0); tr=tr.fillna(med).fillna(0); te=te.fillna(med).fillna(0)
    mean=tr.mean(); std=tr.std().replace(0,1).fillna(1)
    tr=np.clip(((tr-mean)/std).to_numpy(np.float32),-20,20)
    te=np.clip(((te-mean)/std).to_numpy(np.float32),-20,20)
    print("full RDKit descriptors:",tr.shape[1]); return tr,te

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(2048,512),nn.SiLU(),nn.Dropout(.08),nn.Linear(512,128),nn.SiLU())
        self.decoder=nn.Sequential(nn.Linear(128,512),nn.SiLU(),nn.Linear(512,2048))
    def forward(self,x): return self.decoder(self.encoder(x))

def pretrain_encoder(aux_fp):
    seed_all(SEED); model=Autoencoder().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    loader=DataLoader(TensorDataset(torch.from_numpy(aux_fp.astype(np.float32))),batch_size=256,shuffle=True)
    positive=max(float(aux_fp.size/max(aux_fp.sum(),1)-1),1.0)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(positive,25),device=DEVICE))
    for epoch in range(6):
        model.train(); total=0
        for (x,) in loader:
            x=x.to(DEVICE); opt.zero_grad(); loss=loss_fn(model(x),x); loss.backward(); opt.step(); total+=loss.item()*len(x)
        print(f"autoencoder epoch {epoch+1}: {total/len(aux_fp):.5f}")
    return {k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}

class MultiTaskModel(nn.Module):
    def __init__(self,encoder_state,n_tasks,n_desc):
        super().__init__(); ae=Autoencoder(); ae.encoder.load_state_dict(encoder_state); self.encoder=ae.encoder
        self.desc=nn.Sequential(nn.Linear(n_desc,48),nn.SiLU(),nn.LayerNorm(48))
        self.shared=nn.Sequential(nn.Linear(176,192),nn.SiLU(),nn.Dropout(.15),nn.Linear(192,128),nn.SiLU())
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(128,48),nn.SiLU(),nn.Linear(48,1)) for _ in range(n_tasks)])
    def forward(self,fp,desc,task):
        h=self.shared(torch.cat([self.encoder(fp),self.desc(desc)],1)); out=torch.empty(len(h),device=h.device)
        for t,head in enumerate(self.heads):
            mask=task==t
            if mask.any(): out[mask]=head(h[mask]).squeeze(1)
        return out
    def forward_all(self,fp,desc):
        h=self.shared(torch.cat([self.encoder(fp),self.desc(desc)],1))
        return torch.cat([head(h) for head in self.heads],1)

def predict_multitask(model,fp,desc,task,batch=512):
    model.eval(); out=[]
    loader=DataLoader(TensorDataset(torch.from_numpy(fp.astype(np.float32)),torch.from_numpy(desc.astype(np.float32)),torch.from_numpy(task)),batch_size=batch)
    with torch.no_grad():
        for a,b,c in loader: out.append(model(a.to(DEVICE),b.to(DEVICE),c.to(DEVICE)).cpu().numpy())
    return np.concatenate(out)

def predict_multitask_all(model,fp,desc,batch=512):
    model.eval(); out=[]
    loader=DataLoader(TensorDataset(torch.from_numpy(fp.astype(np.float32)),torch.from_numpy(desc.astype(np.float32))),batch_size=batch)
    with torch.no_grad():
        for a,b in loader: out.append(model.forward_all(a.to(DEVICE),b.to(DEVICE)).cpu().numpy())
    return np.vstack(out)

def train_multitask_oof(fp,desc,task,y,groups,encoder_state,test_fp,test_desc,test_task,n_tasks):
    unique=np.array(sorted(set(groups))); splitter=KFold(N_SPLITS,shuffle=True,random_state=SEED)
    oof=np.zeros(len(y),np.float32); all_oof=np.zeros((len(y),n_tasks),np.float32); tests=[]; all_tests=[]
    counts=np.bincount(task,minlength=n_tasks); sample_weights=1/np.maximum(counts[task],1)
    for fold,(gu_tr,gu_va) in enumerate(splitter.split(unique)):
        tr=np.where(np.isin(groups,unique[gu_tr]))[0]; va=np.where(np.isin(groups,unique[gu_va]))[0]
        seed_all(SEED+fold); model=MultiTaskModel(encoder_state,n_tasks,desc.shape[1]).to(DEVICE)
        optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-5)
        sampler=WeightedRandomSampler(sample_weights[tr],num_samples=max(len(tr),4000),replacement=True)
        ds=TensorDataset(torch.from_numpy(fp[tr].astype(np.float32)),torch.from_numpy(desc[tr].astype(np.float32)),
                         torch.from_numpy(task[tr]),torch.from_numpy(y[tr].astype(np.float32)))
        loader=DataLoader(ds,batch_size=128,sampler=sampler); best=1e9; state=None; stale=0
        for epoch in range(100):
            model.train()
            for a,b,c,d in loader:
                a,b,c,d=a.to(DEVICE),b.to(DEVICE),c.to(DEVICE),d.to(DEVICE); optimizer.zero_grad()
                loss=nn.functional.smooth_l1_loss(model(a,b,c),d,beta=.35); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
            vp=predict_multitask(model,fp[va],desc[va],task[va])
            task_losses=[np.mean((vp[task[va]==t]-y[va][task[va]==t])**2)
                         for t in range(n_tasks) if np.any(task[va]==t)]
            val=float(np.mean(task_losses))
            if val<best-1e-4: best=val; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
            else: stale+=1
            if stale>=12: break
        model.load_state_dict(state); all_oof[va]=predict_multitask_all(model,fp[va],desc[va])
        oof[va]=all_oof[va,task[va]]
        test_all=predict_multitask_all(model,test_fp,test_desc); all_tests.append(test_all)
        tests.append(test_all[np.arange(len(test_task)),test_task]); print("multitask fold",fold+1,"epochs",epoch+1)
    return oof,np.mean(tests,axis=0),all_oof,np.mean(all_tests,axis=0)

def cross_property_meta_features(all_predictions, teacher_predictions, tree_prediction, means_array, stds_array):
    raw=all_predictions*stds_array[None,:]+means_array[None,:]
    teacher_raw=teacher_predictions*stds_array[None,:]+means_array[None,:]
    extra=np.column_stack([
        raw[:,3]-raw[:,0],                 # predicted Ei - Eea
        raw[:,1]-raw[:,2],                 # bulk-chain gap difference
        raw[:,5]**2,                       # optical estimate of EPS
        np.sqrt(np.maximum(raw[:,4],0)),   # dielectric estimate of Nc
        teacher_raw[:,3]-teacher_raw[:,0],
        teacher_raw[:,1]-teacher_raw[:,2],
        teacher_raw[:,5]**2,
        np.sqrt(np.maximum(teacher_raw[:,4],0)),
    ])
    return np.column_stack([raw,teacher_raw,extra,tree_prediction])

def nested_meta_oof(X,y,groups,seed):
    kf=GroupKFold(min(5,len(np.unique(groups)))); best=(-1,None,None)
    for alpha in [.1,1.,10.,50.,200.]:
        oof=np.zeros(len(y),np.float32)
        for tr,va in kf.split(X,y,groups):
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
            model.fit(X[tr],y[tr]); oof[va]=model.predict(X[va])
        score=r2_score(y,oof)
        if score>best[0]: best=(score,alpha,oof)
    return best

def scaffold_robust_weights(predictions,y,groups,weights):
    """Prefer fold-stable weights without sacrificing more than 0.002 global R2."""
    matrix=np.column_stack(predictions)
    folds=list(GroupKFold(min(5,len(np.unique(groups)))).split(matrix,y,groups))
    rows=[]
    for weight in weights:
        weight=np.asarray(weight,np.float32); pred=matrix@weight
        global_score=r2_score(y,pred)
        fold_scores=np.array([r2_score(y[va],pred[va]) for _,va in folds],np.float32)
        robust_score=float(np.quantile(fold_scores,.25)-.10*np.std(fold_scores))
        rows.append((global_score,robust_score,weight))
    global_best=max(rows,key=lambda row:row[0])
    eligible=[row for row in rows if row[0]>=global_best[0]-.002]
    robust_best=max(eligible,key=lambda row:row[1])
    selected=.5*global_best[2]+.5*robust_best[2]
    selected_score=r2_score(y,matrix@selected)
    print("robust weights",np.round(selected,3).tolist(),
          "global",round(selected_score,5),"lower-fold",round(robust_best[1],5))
    return selected_score,selected

def tune_xgb_scaffold(X,y,groups,seed,n_trials=8):
    try:
        import optuna
        from xgboost import XGBRegressor
    except Exception as exc:
        try:
            print("installing public Optuna/XGBoost packages...")
            subprocess.check_call([sys.executable,"-m","pip","install","-q","optuna","xgboost"])
            import optuna
            from xgboost import XGBRegressor
        except Exception:
            print("Optuna/XGBoost unavailable:",exc); return None
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds=list(GroupKFold(min(3,len(np.unique(groups)))).split(X,y,groups))
    device="cuda" if torch.cuda.is_available() else "cpu"
    def objective(trial):
        params=dict(n_estimators=trial.suggest_int("n_estimators",300,850),
                    learning_rate=trial.suggest_float("learning_rate",.01,.07,log=True),
                    max_depth=trial.suggest_int("max_depth",2,6),
                    min_child_weight=trial.suggest_float("min_child_weight",1,8),
                    subsample=trial.suggest_float("subsample",.65,1),
                    colsample_bytree=trial.suggest_float("colsample_bytree",.45,.9),
                    reg_alpha=trial.suggest_float("reg_alpha",1e-4,1,log=True),
                    reg_lambda=trial.suggest_float("reg_lambda",.3,5,log=True),
                    objective="reg:squarederror",tree_method="hist",device=device,
                    random_state=seed,n_jobs=-1,verbosity=0)
        oof=np.zeros(len(y),np.float32)
        for tr,va in folds:
            model=XGBRegressor(**params); model.fit(X[tr],y[tr]); oof[va]=model.predict(X[va])
        return r2_score(y,oof)
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective,n_trials=n_trials,show_progress_bar=False)
    params=study.best_params
    params.update(objective="reg:squarederror",tree_method="hist",device=device,
                  random_state=seed,n_jobs=-1,verbosity=0)
    print("Optuna scaffold R2",round(study.best_value,5),"params",params)
    return params

def tree_oof(rich_fp,desc,task,y,groups,test_rich_fp,test_desc,test_task,n_tasks):
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install","-q","lightgbm"])
        from lightgbm import LGBMRegressor
    X=np.hstack([rich_fp,desc]); Xt=np.hstack([test_rich_fp,test_desc])
    oof=np.zeros(len(y),np.float32); pred=np.zeros(len(test_task),np.float32)
    for t in range(n_tasks):
        idx=np.where(task==t)[0]; tidx=np.where(test_task==t)[0]
        local_groups=groups[idx]; kf=GroupKFold(min(5,len(np.unique(local_groups))))
        folds=list(kf.split(X[idx],y[idx],local_groups))
        models=[
            ExtraTreesRegressor(n_estimators=500,max_features=.60,min_samples_leaf=1,random_state=SEED+t,n_jobs=-1),
            LGBMRegressor(n_estimators=650,learning_rate=.02,num_leaves=15 if len(idx)<500 else 31,
                          min_child_samples=5 if len(idx)<500 else 8,colsample_bytree=.75,
                          reg_alpha=.03,reg_lambda=.8,random_state=SEED+t,n_jobs=-1,verbose=-1)]
        model_oofs=[]; model_tests=[]
        for estimator in models:
            moof=np.zeros(len(idx),np.float32); foldtests=[]
            for tr,va in folds:
                model=estimator.__class__(**estimator.get_params()); model.fit(X[idx[tr]],y[idx[tr]])
                moof[va]=model.predict(X[idx[va]]); foldtests.append(model.predict(Xt[tidx]))
            full=estimator.__class__(**estimator.get_params()); full.fit(X[idx],y[idx])
            model_oofs.append(moof); model_tests.append(.75*full.predict(Xt[tidx])+.25*np.mean(foldtests,axis=0))
        best=(-1,0.0)
        for alpha in [0,.25,.5,.75,1.0]:
            candidate=alpha*model_oofs[0]+(1-alpha)*model_oofs[1]; score=r2_score(y[idx],candidate)
            if score>best[0]: best=(score,alpha)
        base_score,alpha=best; chosen_oof=alpha*model_oofs[0]+(1-alpha)*model_oofs[1]
        chosen_test=alpha*model_tests[0]+(1-alpha)*model_tests[1]
        if len(idx)<1200:
            xgb_params=tune_xgb_scaffold(X[idx],y[idx],local_groups,SEED+700+t)
            if xgb_params is not None:
                from xgboost import XGBRegressor
                xoof=np.zeros(len(idx),np.float32); xtests=[]
                for tr,va in folds:
                    model=XGBRegressor(**xgb_params); model.fit(X[idx[tr]],y[idx[tr]])
                    xoof[va]=model.predict(X[idx[va]]); xtests.append(model.predict(Xt[tidx]))
                full=XGBRegressor(**xgb_params); full.fit(X[idx],y[idx])
                xtest=.75*full.predict(Xt[tidx])+.25*np.mean(xtests,axis=0)
                xbest=(base_score,0.0)
                for xweight in [.2,.4,.6,.8,1.0]:
                    score=r2_score(y[idx],(1-xweight)*chosen_oof+xweight*xoof)
                    if score>xbest[0]: xbest=(score,xweight)
                if xbest[0]>=base_score+.005:
                    chosen_oof=(1-xbest[1])*chosen_oof+xbest[1]*xoof
                    chosen_test=(1-xbest[1])*chosen_test+xbest[1]*xtest
                    base_score=xbest[0]
                    print("accepted tuned XGBoost weight",xbest[1])
        oof[idx]=chosen_oof; pred[tidx]=chosen_test
        print("tree ensemble",t,"scaffold R2",round(base_score,5),"ExtraTrees weight",alpha)
    return oof,pred

def global_kernel_oof(binary_fp,task,y,groups,test_binary_fp,test_task,n_tasks):
    """Global Tanimoto SVR for scarce targets; skipped for large Egc/Tg tasks."""
    oof=np.full(len(y),np.nan,np.float32); pred=np.full(len(test_task),np.nan,np.float32)
    scores={}
    for t in range(n_tasks):
        idx=np.where(task==t)[0]; tidx=np.where(test_task==t)[0]
        if len(idx)>1200: continue
        A=binary_fp[idx]; B=test_binary_fp[tidx]
        inter=A@A.T; ai=A.sum(1); union=ai[:,None]+ai[None,:]-inter
        K=np.divide(inter,union,out=np.zeros_like(inter,dtype=np.float32),where=union>0)
        test_inter=B@A.T; bi=B.sum(1); test_union=bi[:,None]+ai[None,:]-test_inter
        Kt=np.divide(test_inter,test_union,out=np.zeros_like(test_inter,dtype=np.float32),where=test_union>0)
        local_groups=groups[idx]; kf=GroupKFold(min(5,len(np.unique(local_groups)))); local=np.zeros(len(idx),np.float32)
        for tr,va in kf.split(K,y[idx],local_groups):
            model=SVR(kernel="precomputed",C=5.0,epsilon=.03,cache_size=4000)
            model.fit(K[np.ix_(tr,tr)],y[idx[tr]]); local[va]=model.predict(K[np.ix_(va,tr)])
        full=SVR(kernel="precomputed",C=5.0,epsilon=.03,cache_size=4000)
        full.fit(K,y[idx]); oof[idx]=local; pred[tidx]=full.predict(Kt)
        scores[t]=r2_score(y[idx],local); print("global kernel",t,"R2",round(scores[t],5))
    return oof,pred

def cross_fitted_tree_teachers(rich_fp,desc,task,y,groups,test_rich_fp,test_desc,n_tasks):
    """Predict every property for every polymer without exposing its scaffold."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install","-q","lightgbm"])
        from lightgbm import LGBMRegressor
    X=np.hstack([rich_fp,desc]); Xt=np.hstack([test_rich_fp,test_desc])
    folds=list(GroupKFold(min(5,len(np.unique(groups)))).split(X,y,groups))
    oof=np.zeros((len(y),n_tasks),np.float32); test=np.zeros((len(Xt),n_tasks),np.float32)
    for source in range(n_tasks):
        source_rows=np.where(task==source)[0]; fold_tests=[]
        for fold,(tr_all,va_all) in enumerate(folds):
            train_mask=np.intersect1d(tr_all,source_rows,assume_unique=False)
            model=LGBMRegressor(n_estimators=700,learning_rate=.02,
                num_leaves=15 if len(source_rows)<500 else 31,
                min_child_samples=5 if len(source_rows)<500 else 8,
                colsample_bytree=.72,reg_alpha=.05,reg_lambda=1.0,
                random_state=SEED+1200+source+fold,n_jobs=-1,verbose=-1)
            model.fit(X[train_mask],y[train_mask]); oof[va_all,source]=model.predict(X[va_all])
            fold_tests.append(model.predict(Xt))
        full=LGBMRegressor(n_estimators=800,learning_rate=.02,
            num_leaves=15 if len(source_rows)<500 else 31,
            min_child_samples=5 if len(source_rows)<500 else 8,
            colsample_bytree=.72,reg_alpha=.05,reg_lambda=1.0,
            random_state=SEED+2200+source,n_jobs=-1,verbose=-1)
        full.fit(X[source_rows],y[source_rows])
        test[:,source]=.75*full.predict(Xt)+.25*np.mean(fold_tests,axis=0)
        own=source_rows
        print("cross-fitted teacher",source,"own-target scaffold R2",round(r2_score(y[own],oof[own,source]),5))
    return oof,test

train_primary=pd.read_csv(find_file("train.csv")); test=pd.read_csv(find_file("test.csv")); aux=pd.read_csv(find_file("PI1M.csv"))
train_primary.target_type=train_primary.target_type.str.lower(); test.target_type=test.target_type.str.lower()
archive_path=find_archive_file("train.csv")
if archive_path is not None:
    archive_train=pd.read_csv(archive_path); archive_train.target_type=archive_train.target_type.str.lower()
    train=pd.concat([train_primary,archive_train],ignore_index=True)
    train=train.groupby(["smiles","target_type"],as_index=False).target.mean()
    print("merged organizer archive:",len(train_primary),"->",len(train),"unique labeled polymers/properties")
else:
    train=train_primary.copy(); print("archive/train.csv not found; using Round 2 labels only")
types=sorted(train.target_type.unique()); type_id={v:i for i,v in enumerate(types)}
task=train.target_type.map(type_id).to_numpy(np.int64); test_task=test.target_type.map(type_id).to_numpy(np.int64)
means=train.groupby("target_type").target.mean().to_dict(); stds=train.groupby("target_type").target.std().to_dict()
y=np.array([(v-means[t])/stds[t] for v,t in zip(train.target,train.target_type)],np.float32)

print("extracting competition features...")
fp,desc=molecular_features(train.smiles); test_fp,test_desc=molecular_features(test.smiles)
print("extracting rich tree fingerprints...")
rich_fp=rich_fingerprints(train.smiles); test_rich_fp=rich_fingerprints(test.smiles)
kernel_fp=rich_fp.copy(); test_kernel_fp=test_rich_fp.copy()
rd_desc,test_rd_desc=full_rdkit_descriptor_pair(train.smiles.tolist(),test.smiles.tolist())
rich_fp=np.hstack([rich_fp,rd_desc]); test_rich_fp=np.hstack([test_rich_fp,test_rd_desc])
dmean=desc.mean(0); dstd=desc.std(0); dstd[dstd<1e-6]=1; desc=(desc-dmean)/dstd; test_desc=(test_desc-dmean)/dstd
print("sampling PI1M for allowed self-supervision...")
aux_sample=aux.sample(min(AUX_SAMPLE,len(aux)),random_state=SEED)["SMILES"].astype(str).tolist()
aux_fp,_=molecular_features(aux_sample,False); encoder_state=pretrain_encoder(aux_fp); del aux_fp,aux

print("generating Murcko scaffold groups...")
groups=train.smiles.astype(str).map(scaffold_group).to_numpy()
nn_oof,nn_test,nn_all_oof,nn_all_test=train_multitask_oof(fp,desc,task,y,groups,encoder_state,test_fp,test_desc,test_task,len(types))
tree_cv,tree_test=tree_oof(rich_fp,desc,task,y,groups,test_rich_fp,test_desc,test_task,len(types))
kernel_cv,kernel_test=global_kernel_oof(kernel_fp,task,y,groups,test_kernel_fp,test_task,len(types))
print("training scaffold-cross-fitted property teachers...")
teacher_oof,teacher_test=cross_fitted_tree_teachers(
    rich_fp,desc,task,y,groups,test_rich_fp,test_desc,len(types))

final=np.zeros(len(test),np.float32)
means_array=np.array([means[name] for name in types],np.float32)
stds_array=np.array([stds[name] for name in types],np.float32)
for name,t in type_id.items():
    idx=np.where(task==t)[0]; tidx=np.where(test_task==t)[0]; best=(-1,None)
    print("\n",name,"neural",r2_score(y[idx],nn_oof[idx]),"tree",r2_score(y[idx],tree_cv[idx]))
    if np.isfinite(kernel_cv[idx]).all():
        two_best=(-1,None)
        two_weights=[]
        for alpha in [0,.2,.4,.6,.8,1.0]:
            two_weights.append((alpha,1-alpha,0.0))
            score=r2_score(y[idx],alpha*nn_oof[idx]+(1-alpha)*tree_cv[idx])
            if score>two_best[0]: two_best=(score,alpha)
        best=(-1,None); all_weights=[]
        for neural_w in [0,.2,.4,.6,.8,1.0]:
            for tree_w in [0,.2,.4,.6,.8,1.0]:
                if neural_w+tree_w>1.0001: continue
                kernel_w=1-neural_w-tree_w
                all_weights.append((neural_w,tree_w,kernel_w))
                candidate=neural_w*nn_oof[idx]+tree_w*tree_cv[idx]+kernel_w*kernel_cv[idx]
                score=r2_score(y[idx],candidate)
                if score>best[0]: best=(score,(neural_w,tree_w,kernel_w))
        if best[0]<two_best[0]+.003:
            all_weights=two_weights
        robust_score,robust_weight=scaffold_robust_weights(
            [nn_oof[idx],tree_cv[idx],kernel_cv[idx]],y[idx],groups[idx],all_weights)
        best=(robust_score,tuple(robust_weight))
        neural_w,tree_w,kernel_w=best[1]
        baseline_oof=neural_w*nn_oof[idx]+tree_w*tree_cv[idx]+kernel_w*kernel_cv[idx]
        baseline_test=neural_w*nn_test[tidx]+tree_w*tree_test[tidx]+kernel_w*kernel_test[tidx]
        weight_text=f"weights neural/tree/kernel={neural_w:.1f}/{tree_w:.1f}/{kernel_w:.1f}"
    else:
        two_weights=[]
        for alpha in [0,.2,.4,.6,.8,1.0]:
            two_weights.append((alpha,1-alpha))
            score=r2_score(y[idx],alpha*nn_oof[idx]+(1-alpha)*tree_cv[idx])
            if score>best[0]: best=(score,alpha)
        robust_score,robust_weight=scaffold_robust_weights(
            [nn_oof[idx],tree_cv[idx]],y[idx],groups[idx],two_weights)
        best=(robust_score,robust_weight[0]); alpha=best[1]
        baseline_oof=alpha*nn_oof[idx]+(1-alpha)*tree_cv[idx]
        baseline_test=alpha*nn_test[tidx]+(1-alpha)*tree_test[tidx]
        weight_text=f"neural weight={alpha:.1f}"
    train_tree_raw=tree_cv[idx]*stds[name]+means[name]; test_tree_raw=tree_test[tidx]*stds[name]+means[name]
    meta_X=cross_property_meta_features(nn_all_oof[idx],teacher_oof[idx],train_tree_raw,means_array,stds_array)
    meta_Xt=cross_property_meta_features(nn_all_test[tidx],teacher_test[tidx],test_tree_raw,means_array,stds_array)
    meta_score,meta_alpha,meta_oof=nested_meta_oof(meta_X,y[idx],groups[idx],SEED+500+t)
    print(name,"blend R2",best[0],weight_text,"cross-property R2",meta_score,"ridge",meta_alpha)
    if meta_score>best[0]+.003:
        meta_model=make_pipeline(StandardScaler(),Ridge(alpha=meta_alpha)); meta_model.fit(meta_X,y[idx])
        normalized=.85*meta_model.predict(meta_Xt)+.15*baseline_test
    else:
        normalized=baseline_test
    final[tidx]=normalized*stds[name]+means[name]

known=test[["smiles","target_type"]].merge(train[["smiles","target_type","target"]],on=["smiles","target_type"],how="left")
known_mask=known.target.notna().to_numpy()
final[known_mask]=known.loc[known_mask,"target"].to_numpy()
print("exact organizer-provided matches:",int(known_mask.sum()),"/",len(test))

# Recover equivalent atom-order SMILES, but reject canonical collisions with inconsistent labels.
canonical_train=train[["smiles","target_type","target"]].copy()
canonical_train["canonical"]=canonical_train.smiles.map(canonical_key)
canonical_stats=(canonical_train[canonical_train.canonical.ne("")]
    .groupby(["canonical","target_type"]).target.agg(["mean","min","max"]).reset_index())
canonical_stats["tolerance"]=canonical_stats.target_type.map(stds).astype(float)*.02
canonical_stats=canonical_stats[(canonical_stats["max"]-canonical_stats["min"])<=canonical_stats.tolerance]
canonical_test=test[["smiles","target_type"]].copy()
canonical_test["canonical"]=canonical_test.smiles.map(canonical_key)
canonical_known=canonical_test.merge(
    canonical_stats[["canonical","target_type","mean"]],on=["canonical","target_type"],how="left")
canonical_mask=(~known_mask)&canonical_known["mean"].notna().to_numpy()
final[canonical_mask]=canonical_known.loc[canonical_mask,"mean"].to_numpy()
print("additional collision-safe canonical matches:",int(canonical_mask.sum()))
all_known=known_mask|canonical_mask
print(pd.Series(test.loc[all_known,"target_type"]).value_counts().to_dict())
submission=test[["id"]].copy(); submission["target"]=final; submission=submission.sort_values("id").reset_index(drop=True)
submission.to_csv("submission.csv",index=False); print("\nSaved submission.csv\n",submission.head())
