# Full FSD main: fresh review accepted

2026-09-13. Master archives the completed read-only review returned by /root/fsd_full_main_fresh_review (Astra high). This is a faithful decision/evidence summary, not a new numerical run or a claim of human peer review.

ACCEPT; execution valid with the historical resource exceptions below. Scientific valid_positive is limited to fixed representations, five-second reset contexts and unified ridge readout on FSD200 clip classification. Deduplication effects remain valid_inconclusive; CNN6 and BEATs T4-minus-G are valid negative results. No scientific repair or retraining required.

Binding: execution_04/MAIN_PACKET.json SHA256 08e927124469e1682c34f8f44ae067508ce3d9942a35093013419bfde893d1ad. Terminal receipt SHA256 78aa91157ef555e2d613d4e2620bf798c1cc3fc4315f53866237161cce42196f. Current FSD coordination nonce fsd-full-main-user-resume-20260912-01; coordination terminal SHA256 8b94d9bd3bfdb8cd9785780d3f09d03dd2fe40d24e17999fc99630bb2f4ebffb. Original author acceptance flags remain unchanged; this external review records acceptance.

Independent checks: all 51,197 IDs and labels, official 36,796/4,170/10,231 split; 106,158 complete windows; longest ID124858 has111 windows. Four families/204,788 feature-row hashes and18 caches checked. All36 train-only normalization and target means checked (max1.60e-14),180 validation candidates and selected lambda reproduced. All36 test per-class/macro/micro AP plus E1 source exclusion reproduced (max5.55e-16);396 selected-model predictions reconstructed (max5.69e-16). All selections locked before first test, with one successful readout and test session per condition. Reconstructed2,297 uploader/content groups;108 complete200-class bootstrap resamples (draws0/317/1999 percondition) matched within1.11e-16; all36x2000 saved draws contain200 classes. All76 paired intervals and author RESULT_TABLE agree. Bootstrap was sampled independently, not exhaustively regenerated. Raw audio/features were not re-extracted.

Official test macro200 AP:

| Representation | G full | G dedup | T4 full | T4 dedup |
|---|---:|---:|---:|---:|
| logmel | .076103 | .076097 | .089549 | .089554 |
| E1 z | .110619 | .110614 | .124711 | .124735 |
| E1 q2 | .087100 | .087103 | .099606 | .099599 |
| E1 q4 | .095509 | .095504 | .109092 | .109092 |
| CNN6 | .441266 | .441283 | .427592 | .427605 |
| EnCodec z | .144143 | .144159 | .149121 | .149116 |
| EnCodec q2 | .121578 | .121574 | .125564 | .125588 |
| EnCodec q4 | .130125 | .130141 | .135186 | .135222 |
| BEATs | .601386 | .601406 | .591403 | .591400 |

Full/G BEATs-CNN6 +.160120, paired95% interval [.150303,.164919]. q4-q2 is positive in E1 and EnCodec in both pooling and training arms; full/G E1 +.008409 [.006843,.010357], EnCodec +.008547 [.006218,.010891]. This is readout depth, not ALL-EARLY pressure. T4-G is positive for logmel/E1/EnCodec but negative for CNN6 (-.013674 [-.018049,-.009969]) and BEATs (-.009983 [-.013286,-.006986]) in full arms. T4 changes dimension and reader capacity too, so time retention alone is not identified. Removing the two predefined train duplicates changes18 scores by at most .00003676; all intervals cross zero, not proof of equivalence. E1 source panel excludes119 test files and retains10112, not proof of no other exposure.

Formal recorded stage seconds: features_eight39247.178, BEATs42898.856, readouts419.899, test156.626, uncertainty/contrasts4519.341; total87241.899s/24.234h. Including core/pilot all136 sessions87874.908s/24.410h. Separate external-memory wait17480.156s;2/2 low-memory retries used. Costs include failures/pauses/retries and remain within authorized cumulative limits; batch6.183GB, minimum recorded D free11.988GB.

Resource exception: resources.jsonl contains101 available-memory samples below1GiB, minimum334913536B (including PID25000/features_eight). FULL_FEATURE_FAILURE_1789198005.json and execution_04/RECOVERY_LEDGER.json preserve low-memory stop and recoveries. Do not claim uninterrupted reserve compliance. No evidence that these exceptions damaged saved data or invalidate numerical comparisons; do not rerun successful science to erase them.

Limits: fixed-model descriptive pointwise intervals exclude training randomness and simultaneous multiple-comparison guarantees. EnCodec FSD pretraining, CNN6/BEATs AudioSet exposure and E1 ESC/FSD source overlap require disclosure. This does not establish SED localization, pressure effects, waveform quality or completion of C1-C5. Full-audio coverage is established; native uninterrupted context equivalence and context30 sensitivity remain OPEN and mandatory.

Review ran two main numerical processes totaling149.864s, peak237.977MiB; other lightweight checks kept total below360s. No Torch, audio, training, downloads, writes or experimental process operations; read-only responsibility was not claimed to be OS-enforced.
