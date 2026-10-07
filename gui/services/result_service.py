from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable
from basil_core.protocol_compatibility import normalize_config

@dataclass(slots=True)
class ResultRecord:
    path: Path; name: str; status: str; dataset: str; approach: str; split: str; noise: float | None; mitigation: str
    final_average: float | None=None; final_worst: float | None=None; completed_at: str=""; error: str=""
    protocol: str=""; research_valid: bool=True; purpose: str=""; noise_semantics: str=""

class ResultService:
    def __init__(self,root: Path | str,additional_roots=()): self.root=Path(root); self.roots=(self.root,*(Path(p) for p in additional_roots)); self._records: list[ResultRecord]|None=None
    def discover(self,refresh: bool=False) -> list[ResultRecord]:
        if self._records is not None and not refresh: return list(self._records)
        records=[]
        for path in (p for root in self.roots if root.exists() for p in root.rglob("run.json")):
            try:
                data=json.loads(path.read_text(encoding="utf-8")); config=normalize_config(data.get("config",{}))
                research=bool(config.get("experimentProtocol"));semantics=config.get("channelNoiseSemantics","")
                sigma=config.get("channelNoiseSigmaRelative",config.get("channelNoiseSigmaRel",0)) if semantics=="relative_l2_gaussian" and research else config.get("channelNoiseSigmaAbsolute",0) if research else config.get("channelNoiseSigma",0)
                ebm=config.get("ebmMode","none")!="none";ss=config.get("snapshotSelection",False)
                mitigation="ss_ebm" if ss and ebm else "ss" if ss else "ebm" if ebm else "none"
                records.append(ResultRecord(path,str(config.get("experimentName",path.parent.name)),str(data.get("status","unknown")),str(config.get("dataset","unknown")),str(config.get("aggregationMode","unknown") if research else config.get("approach","unknown")),str(config.get("partitionStrategy") if research else config.get("split","nonIID" if config.get("nonIID") else "IID")),float(sigma) if config.get("useChannelNoise") else None,mitigation if research else str(config.get("mitigation",config.get("noiseMitigation","none"))),data.get("finalAverageAccuracy"),data.get("finalWorstAccuracy"),str(data.get("completedAt",data.get("startedAt",""))),protocol=config.get("experimentProtocol",""),research_valid=bool(data.get("researchValid",True)),purpose=config.get("purpose",""),noise_semantics=semantics))
            except (OSError,json.JSONDecodeError,TypeError,ValueError) as error:
                records.append(ResultRecord(path,path.parent.name,"malformed","unknown","unknown","unknown",None,"unknown",error=str(error)))
        self._records=sorted(records,key=lambda item:item.completed_at,reverse=True); return list(self._records)
    def filter(self,records: Iterable[ResultRecord],*,search: str="",dataset: str="",approach: str="",status: str="",split: str="",mitigation: str="",noise: str="") -> list[ResultRecord]:
        needle=search.casefold()
        return [r for r in records if (not needle or needle in r.name.casefold()) and (not dataset or r.dataset==dataset) and (not approach or r.approach==approach) and (not status or r.status==status) and (not split or r.split==split) and (not mitigation or r.mitigation==mitigation) and (not noise or (noise=="off" and r.noise is None) or (r.noise is not None and f"{r.noise:g}"==noise))]
    def load_metrics(self,record: ResultRecord) -> dict[str,Any]:
        import numpy as np
        data=json.loads(record.path.read_text(encoding="utf-8")); candidate=data.get("artifactPaths",{}).get("metrics")
        local=record.path.with_name("metrics.npz")
        path=local if local.exists() else (Path(candidate) if candidate else local)
        if not path.is_absolute() and not path.exists(): path=Path.cwd()/path
        if not path.exists(): return {}
        with np.load(path,allow_pickle=False) as metrics: return {name:metrics[name].copy() for name in metrics.files}
    def telemetry_path(self,record: ResultRecord) -> Path|None:
        try: candidate=json.loads(record.path.read_text(encoding="utf-8")).get("artifactPaths",{}).get("telemetry")
        except (OSError,json.JSONDecodeError): return None
        local=record.path.with_name("activation_telemetry.json.gz" if record.protocol else "telemetry.npz")
        path=local if local.exists() else Path(candidate) if candidate else local
        if not path.is_absolute() and not path.exists(): path=Path.cwd()/path
        return path if path.exists() else None
