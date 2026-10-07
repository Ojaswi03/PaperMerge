import csv
import gzip
import json

import numpy as np


def test_full_research_plot_pipeline_uses_all_measured_rounds(tmp_path,monkeypatch):
    import basil_core.iid_study as study
    from reporting.iid_study_plots import generate_run_plots,generate_comparison
    plots=tmp_path/"plots"
    monkeypatch.setattr(study,"PLOT_ROOT",plots)
    directories=[]
    for index in range(2):
        source=tmp_path/f"test_{index}";source.mkdir();directories.append(source)
        manifest=dict(status="completed",researchValid=True,nRounds=50,attackStartRound=20,
            experimentName=f"Fixture {index}",iidPartitionHash="same",initialModelHash="same",
            config={"useChannelNoise":bool(index)})
        (source/"run.json").write_text(json.dumps(manifest))
        curve=np.linspace(.1,.5,50)
        np.savez(source/"metrics.npz",fullTestAccuracy=curve,averageAccuracy=curve-.01,
            worstNodeAccuracy=curve-.02,fullTestLoss=np.linspace(2.3,1.2,50),
            fullTestPerClassAccuracy=np.tile(curve[:,None],(1,10)),
            roundNodeAccuracy=np.tile(curve[:,None],(1,10)))
        with gzip.open(source/"activation_telemetry.json.gz","wt") as handle: json.dump([],handle)
        keys=("honest_selection_rate","byzantine_selection_rate","attacked_source_selection_rate","mean_snapshot_age","mean_local_ce",
            "ebm_penalty","ordinary_gradient_norm","ebm_correction_norm","ebm_correction_ratio",
            "channel_noise_norm","noise_to_model_norm_ratio")
        with (source/"round_metrics.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
            writer.writerows({k:.1 for k in keys} for _ in range(50))
        output=plots/f"test_{index}"
        count=generate_run_plots(source,output)
        assert count>=20
        assert (output/"full_test_accuracy_vs_round.png").is_file()
        assert (output/"per_class_accuracy_vs_round.png").is_file()
        if index: assert (output/"ebm_correction_ratio_vs_round.png").is_file()
    generate_comparison(directories,plots/"comparison")
    assert (plots/"comparison/full_test_accuracy_comparison.png").is_file()
    assert (plots/"comparison/attacked_source_selection_rate_comparison.png").is_file()
