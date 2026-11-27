window.DASH = {
 "built": "2026-11-27 14:36",
 "tests": {
  "passed": 128,
  "failed": 0,
  "summary": "128 passed, 1 warning in 26.40s"
 },
 "rooms": {
  "classroom_01": {
   "windows": 10049,
   "low": 1367
  },
  "room_B": {
   "windows": 8699,
   "low": 4392
  },
  "room_C": {
   "windows": 4130,
   "low": 807
  }
 },
 "human_windows": 22878,
 "human_low": 6566,
 "final_model": {
  "name": "Class-relative motion + hard-evidence floor",
  "checkpoint": "checkpoints/motion_rule_v1.json",
  "coef": {
   "a": 0.0515,
   "b": -0.5479,
   "threshold": 0.5145
  },
  "unseen": {
   "balanced_accuracy": 0.5926,
   "accuracy": 0.5689,
   "macro_f1": 0.5222,
   "roc_auc": 0.6217,
   "low_recall": 0.6135,
   "always_attentive_accuracy": 0.7247
  },
  "per_room": {
   "classroom_01": {
    "balacc": 0.5688,
    "acc": 0.3914,
    "macro_f1": 0.371,
    "auc": 0.5951,
    "low_recall": 0.8089,
    "always_high": 0.8693,
    "n": 3444.0
   },
   "room_B": {
    "balacc": 0.5452,
    "acc": 0.5451,
    "macro_f1": 0.5451,
    "auc": 0.5648,
    "low_recall": 0.5385,
    "always_high": 0.4921,
    "n": 2585.0
   },
   "room_C": {
    "balacc": 0.6637,
    "acc": 0.7703,
    "macro_f1": 0.6505,
    "auc": 0.7053,
    "low_recall": 0.4931,
    "always_high": 0.8126,
    "n": 1537.0
   }
  },
  "live": {
   "clip_004": {
    "auc": 0.625,
    "balanced_accuracy": 0.571,
    "accuracy": 0.715,
    "always_high": 0.789,
    "matched": 1305
   },
   "clip_001": {
    "auc": 0.681,
    "balanced_accuracy": 0.578,
    "accuracy": 0.766,
    "always_high": 0.919,
    "matched": 1531
   },
   "11": {
    "auc": 0.777,
    "balanced_accuracy": 0.74,
    "accuracy": 0.673,
    "always_high": 0.848,
    "matched": 1168
   },
   "non_attentive_1": {
    "auc": 0.399,
    "balanced_accuracy": 0.475,
    "accuracy": 0.536,
    "always_high": 0.123,
    "matched": 1199
   },
   "test_8": {
    "auc": 0.84,
    "balanced_accuracy": 0.777,
    "accuracy": 0.767,
    "always_high": 0.813,
    "matched": 1512
   }
  },
  "live_mean": {
   "auc": 0.6644,
   "balanced_accuracy": 0.6282,
   "accuracy": 0.6914
  }
 },
 "bigru_final": {
  "checkpoint": "checkpoints\\final_human_v1.pt",
  "sha256": "0d4ad13234eb2dce",
  "threshold": 0.8800000000000001,
  "masked": [
   0,
   1,
   2,
   3,
   4,
   21
  ]
 },
 "old_claims": [
  {
   "claim": "Clip 001: 95.13% accuracy",
   "truth": "Always answering HIGH scores 95.22%; the model found 0 of 51 LOW cases",
   "fixed": "Macro-F1 48.75%, balanced accuracy 49.95%"
  },
  {
   "claim": "Clip 001: 97.50% macro-F1",
   "truth": "That was the F1 of the HIGH class only",
   "fixed": "True macro-F1 48.75%"
  },
  {
   "claim": "Clip 007: 80.65% accuracy on held-out data",
   "truth": "Clip 007 labels are the clip_004 training labels (data leakage)",
   "fixed": "Removed as a test set"
  },
  {
   "claim": "87.22% on 4,386 held-out students' sequences",
   "truth": "Matrix sums to 4,602; 2/3 of labels were made by an older model; same people under different IDs",
   "fixed": "Rebuilt with human labels and person/classroom splits"
  },
  {
   "claim": "Video Project 3: 90.22% vs ground truth",
   "truth": "No human labels exist; it was agreement with the rule-based rubric",
   "fixed": "Shown only as an unlabelled demo"
  },
  {
   "claim": "Threshold 0.45 'validated'",
   "truth": "Chosen on the human test annotation",
   "fixed": "Threshold chosen on held-out persons only"
  }
 ],
 "person_cv": [
  {
   "model": "Always HIGH",
   "acc": 0.864,
   "balacc": 0.5,
   "macro_f1": 0.4635,
   "low_recall": 0.0,
   "auc": 0.5,
   "auc_ci": null
  },
  {
   "model": "BiGRU (old recipe)",
   "acc": 0.7913,
   "balacc": 0.5193,
   "macro_f1": 0.5202,
   "low_recall": 0.1456,
   "auc": 0.6049,
   "auc_ci": [
    0.4626,
    0.7635
   ]
  },
  {
   "model": "Logistic regression",
   "acc": 0.7168,
   "balacc": 0.5896,
   "macro_f1": 0.5542,
   "low_recall": 0.4148,
   "auc": 0.6579,
   "auc_ci": [
    0.4775,
    0.8127
   ]
  }
 ],
 "person_cv_info": {
  "windows": 10049,
  "persons": 23,
  "persons_low": 10,
  "non_overlapping": 697
 },
 "cross_all": [
  {
   "room": "classroom_01",
   "model": "BiGRU",
   "acc": 0.1914,
   "balacc": 0.3776,
   "macro_f1": 0.1911,
   "low_recall": 0.6335,
   "auc": 0.3166,
   "auc_ci": [
    0.1625,
    0.5188
   ],
   "always_high": 0.864
  },
  {
   "room": "classroom_01",
   "model": "Logistic regression",
   "acc": 0.2299,
   "balacc": 0.4853,
   "macro_f1": 0.2299,
   "low_recall": 0.8361,
   "auc": 0.4356,
   "auc_ci": [
    0.2865,
    0.6331
   ],
   "always_high": 0.864
  },
  {
   "room": "classroom_01",
   "model": "Old deployed model",
   "acc": 0.8834,
   "balacc": 0.7593,
   "macro_f1": 0.7555,
   "low_recall": 0.5889,
   "auc": 0.8356,
   "auc_ci": null,
   "always_high": 0.864,
   "contaminated": true
  },
  {
   "room": "room_B",
   "model": "BiGRU",
   "acc": 0.6248,
   "balacc": 0.6269,
   "macro_f1": 0.6079,
   "low_recall": 0.413,
   "auc": 0.5847,
   "auc_ci": [
    0.4581,
    0.7014
   ],
   "always_high": 0.4951
  },
  {
   "room": "room_B",
   "model": "Logistic regression",
   "acc": 0.6127,
   "balacc": 0.6127,
   "macro_f1": 0.6127,
   "low_recall": 0.6122,
   "auc": 0.5783,
   "auc_ci": [
    0.4119,
    0.7146
   ],
   "always_high": 0.4951
  },
  {
   "room": "room_B",
   "model": "Old deployed model",
   "acc": 0.369,
   "balacc": 0.3662,
   "macro_f1": 0.3096,
   "low_recall": 0.656,
   "auc": 0.3264,
   "auc_ci": null,
   "always_high": 0.4951,
   "contaminated": false
  },
  {
   "room": "room_C",
   "model": "BiGRU",
   "acc": 0.4438,
   "balacc": 0.4728,
   "macro_f1": 0.4097,
   "low_recall": 0.5204,
   "auc": 0.4289,
   "auc_ci": [
    0.2562,
    0.6356
   ],
   "always_high": 0.8046
  },
  {
   "room": "room_C",
   "model": "Logistic regression",
   "acc": 0.6174,
   "balacc": 0.4817,
   "macro_f1": 0.4784,
   "low_recall": 0.259,
   "auc": 0.4375,
   "auc_ci": [
    0.3055,
    0.5897
   ],
   "always_high": 0.8046
  },
  {
   "room": "room_C",
   "model": "Old deployed model",
   "acc": 0.34,
   "balacc": 0.4805,
   "macro_f1": 0.3374,
   "low_recall": 0.7113,
   "auc": 0.4593,
   "auc_ci": null,
   "always_high": 0.8046,
   "contaminated": false
  }
 ],
 "cross_final": [
  {
   "room": "classroom_01",
   "model": "BiGRU",
   "acc": 0.8372,
   "balacc": 0.5468,
   "macro_f1": 0.5537,
   "low_recall": 0.1478,
   "auc": 0.5848,
   "auc_ci": [
    0.3829,
    0.7797
   ],
   "always_high": 0.864
  },
  {
   "room": "classroom_01",
   "model": "Logistic regression",
   "acc": 0.1681,
   "balacc": 0.5185,
   "macro_f1": 0.159,
   "low_recall": 1.0,
   "auc": 0.6301,
   "auc_ci": [
    0.4744,
    0.789
   ],
   "always_high": 0.864
  },
  {
   "room": "classroom_01",
   "model": "Old deployed model",
   "acc": 0.8834,
   "balacc": 0.7593,
   "macro_f1": 0.7555,
   "low_recall": 0.5889,
   "auc": 0.8356,
   "auc_ci": null,
   "always_high": 0.864,
   "contaminated": true
  },
  {
   "room": "room_B",
   "model": "BiGRU",
   "acc": 0.4786,
   "balacc": 0.4773,
   "macro_f1": 0.4694,
   "low_recall": 0.6043,
   "auc": 0.5563,
   "auc_ci": [
    0.408,
    0.6856
   ],
   "always_high": 0.4951
  },
  {
   "room": "room_B",
   "model": "Logistic regression",
   "acc": 0.5552,
   "balacc": 0.5539,
   "macro_f1": 0.5461,
   "low_recall": 0.6903,
   "auc": 0.6284,
   "auc_ci": [
    0.499,
    0.7404
   ],
   "always_high": 0.4951
  },
  {
   "room": "room_B",
   "model": "Old deployed model",
   "acc": 0.369,
   "balacc": 0.3662,
   "macro_f1": 0.3096,
   "low_recall": 0.656,
   "auc": 0.3264,
   "auc_ci": null,
   "always_high": 0.4951,
   "contaminated": false
  },
  {
   "room": "room_C",
   "model": "BiGRU",
   "acc": 0.2838,
   "balacc": 0.501,
   "macro_f1": 0.2819,
   "low_recall": 0.8575,
   "auc": 0.4649,
   "auc_ci": [
    0.3242,
    0.5799
   ],
   "always_high": 0.8046
  },
  {
   "room": "room_C",
   "model": "Logistic regression",
   "acc": 0.3182,
   "balacc": 0.5097,
   "macro_f1": 0.3181,
   "low_recall": 0.824,
   "auc": 0.5292,
   "auc_ci": [
    0.4134,
    0.6563
   ],
   "always_high": 0.8046
  },
  {
   "room": "room_C",
   "model": "Old deployed model",
   "acc": 0.34,
   "balacc": 0.4805,
   "macro_f1": 0.3374,
   "low_recall": 0.7113,
   "auc": 0.4593,
   "auc_ci": null,
   "always_high": 0.8046,
   "contaminated": false
  }
 ],
 "pooled": {
  "all_channels": {
   "n": 22878.0,
   "accuracy": 0.4017,
   "balanced_accuracy": 0.4228,
   "macro_f1": 0.3913,
   "low_recall": 0.4721,
   "low_precision": 0.2327,
   "low_f1": 0.3118,
   "high_f1": 0.4709,
   "roc_auc": 0.4228,
   "always_attentive_accuracy": 0.713,
   "roc_auc_within_room_mean": 0.4434
  },
  "no_direction": {
   "n": 22878.0,
   "accuracy": 0.6009,
   "balanced_accuracy": 0.5828,
   "macro_f1": 0.5641,
   "low_recall": 0.5404,
   "low_precision": 0.3673,
   "low_f1": 0.4373,
   "high_f1": 0.6908,
   "roc_auc": 0.5828,
   "always_attentive_accuracy": 0.713,
   "roc_auc_within_room_mean": 0.5354
  }
 },
 "flips": [
  {
   "video": "clip_004",
   "Facing the camera": 0.3266,
   "Head yaw": 0.2495,
   "Gaze pitch": 0.4974,
   "Torso lean": 0.5739
  },
  {
   "video": "clip_001",
   "Facing the camera": 0.2644,
   "Head yaw": 0.2619,
   "Gaze pitch": 0.5264,
   "Torso lean": 0.6069
  },
  {
   "video": "11",
   "Facing the camera": 0.733,
   "Head yaw": 0.6981,
   "Gaze pitch": 0.7319,
   "Torso lean": 0.5852
  },
  {
   "video": "non_attentive_1",
   "Facing the camera": 0.4049,
   "Head yaw": 0.6844,
   "Gaze pitch": 0.5668,
   "Torso lean": 0.6671
  },
  {
   "video": "test_8",
   "Facing the camera": 0.554,
   "Head yaw": 0.4062,
   "Gaze pitch": 0.499,
   "Torso lean": 0.6143
  }
 ],
 "improvement": {
  "n_variants": 40,
  "top": [
   {
    "name": "32-D no camera-relative | hgb | raw | 5s seg",
    "balacc": 0.5822,
    "auc": 0.6005
   },
   {
    "name": "32-D no camera-relative | hgb | raw | window",
    "balacc": 0.5637,
    "auc": 0.6099
   },
   {
    "name": "32-D no camera-relative | logreg | raw | 5s seg",
    "balacc": 0.544,
    "auc": 0.6227
   },
   {
    "name": "gaze yaw/pitch + deviation from class | video-norm | 5s seg",
    "balacc": 0.5436,
    "auc": 0.4917
   },
   {
    "name": "32-D no camera-relative | hgb | video-norm | 5s seg",
    "balacc": 0.5424,
    "auc": 0.641
   }
  ],
  "calibrated": {
   "30%": {
    "balacc": 0.7082,
    "acc": 0.8071,
    "macro_f1": 0.6614,
    "low_recall": 0.6577,
    "always_high": 0.705,
    "auc": 0.7225,
    "per_video": {
     "11": 0.6941,
     "clip_001": 0.7106,
     "clip_004": 0.7278,
     "non_attentive_1": 0.6307,
     "test_8": 0.778
    }
   },
   "50%": {
    "balacc": 0.7113,
    "acc": 0.7928,
    "macro_f1": 0.6376,
    "low_recall": 0.6848,
    "always_high": 0.7174,
    "auc": 0.7451,
    "per_video": {
     "11": 0.7084,
     "clip_001": 0.7776,
     "clip_004": 0.7739,
     "non_attentive_1": 0.748,
     "test_8": 0.5486
    }
   }
  },
  "identity": {
   "11": {
    "unseen_students_auc": 0.296,
    "same_students_later_auc": 0.684
   },
   "clip_001": {
    "unseen_students_auc": 0.451,
    "same_students_later_auc": 0.788
   },
   "clip_004": {
    "unseen_students_auc": 0.512,
    "same_students_later_auc": 0.923
   },
   "non_attentive_1": {
    "unseen_students_auc": 0.566,
    "same_students_later_auc": 0.832
   },
   "test_8": {
    "unseen_students_auc": 0.511,
    "same_students_later_auc": 0.951
   }
  },
  "bigru_seeds": {
   "pooled_balacc": [
    0.612,
    0.018
   ],
   "pooled_acc": [
    0.635,
    0.029
   ],
   "mean_room_balacc": [
    0.523,
    0.023
   ],
   "mean_room_auc": [
    0.531,
    0.014
   ]
  },
  "daisee": {
   "no_daisee": {
    "mean_room_balacc": 0.5272,
    "mean_room_auc": 0.5091
   },
   "daisee_init": {
    "mean_room_balacc": 0.5446,
    "mean_room_auc": 0.554
   },
   "daisee_init+mix": {
    "mean_room_balacc": 0.5462,
    "mean_room_auc": 0.5437
   }
  },
  "datasets": [
   {
    "name": "Motion + study protection (final)",
    "balacc": 0.6177,
    "auc": 0.6477
   },
   {
    "name": "Motion only",
    "balacc": 0.6016,
    "auc": 0.6486
   },
   {
    "name": "Motion + rubric",
    "balacc": 0.55,
    "auc": 0.5156
   },
   {
    "name": "Motion + BiGRU",
    "balacc": 0.5874,
    "auc": 0.6228
   },
   {
    "name": "BiGRU alone",
    "balacc": 0.5858,
    "auc": 0.5723
   },
   {
    "name": "Motion + DAiSEE individual",
    "balacc": 0.5882,
    "auc": 0.6038
   },
   {
    "name": "Motion + SCB behaviour",
    "balacc": 0.5031,
    "auc": 0.5661
   },
   {
    "name": "Motion + DAiSEE + SCB",
    "balacc": 0.4954,
    "auc": 0.5431
   },
   {
    "name": "DAiSEE individual alone",
    "balacc": 0.5335,
    "auc": 0.4767
   },
   {
    "name": "SCB behaviour alone",
    "balacc": 0.4669,
    "auc": 0.3883
   }
  ]
 },
 "ablation": [
  {
   "variant": "Raw (relative to camera)",
   "classroom_01": 0.4356,
   "room_B": 0.5783,
   "room_C": 0.4375,
   "mean": 0.4838
  },
  {
   "variant": "Relative to the student's usual direction",
   "classroom_01": 0.4436,
   "room_B": 0.5512,
   "room_C": 0.5975,
   "mean": 0.5308
  },
  {
   "variant": "Relative to the whole class's direction",
   "classroom_01": 0.4016,
   "room_B": 0.5318,
   "room_C": 0.5034,
   "mean": 0.4789
  },
  {
   "variant": "Head/gaze direction removed",
   "classroom_01": 0.6301,
   "room_B": 0.6284,
   "room_C": 0.5292,
   "mean": 0.5959
  }
 ],
 "demos": [
  {
   "key": "classroom_01_clip001",
   "title": "Computer-lab classroom (clip 001)",
   "room": "classroom_01",
   "labelled": true,
   "video": "media/classroom_01_clip001.mp4",
   "poster": "media/classroom_01_clip001.jpg",
   "frame": [
    832,
    464
   ],
   "duration": 9.91,
   "students_tracked": 22,
   "timeline": [
    [
     1.7,
     49.1,
     54.5,
     11
    ],
    [
     1.8,
     49.3,
     65.0,
     20
    ],
    [
     1.9,
     49.2,
     65.0,
     20
    ],
    [
     2.0,
     49.1,
     60.0,
     20
    ],
    [
     2.1,
     48.8,
     60.0,
     20
    ],
    [
     2.2,
     48.6,
     65.0,
     20
    ],
    [
     2.3,
     48.7,
     65.0,
     20
    ],
    [
     2.4,
     48.9,
     60.0,
     20
    ],
    [
     2.5,
     49.3,
     55.0,
     20
    ],
    [
     2.6,
     49.5,
     65.0,
     20
    ],
    [
     2.7,
     50.3,
     52.6,
     19
    ],
    [
     2.8,
     50.8,
     42.1,
     19
    ],
    [
     2.9,
     51.0,
     36.8,
     19
    ],
    [
     3.0,
     49.7,
     42.1,
     19
    ],
    [
     3.1,
     50.7,
     36.8,
     19
    ],
    [
     3.2,
     50.0,
     36.8,
     19
    ],
    [
     3.3,
     51.2,
     31.6,
     19
    ],
    [
     3.4,
     51.6,
     31.6,
     19
    ],
    [
     3.5,
     50.1,
     36.8,
     19
    ],
    [
     3.6,
     52.3,
     26.3,
     19
    ],
    [
     3.7,
     53.1,
     21.1,
     19
    ],
    [
     3.8,
     54.0,
     15.8,
     19
    ],
    [
     3.9,
     54.5,
     15.8,
     19
    ],
    [
     4.0,
     55.1,
     10.5,
     19
    ],
    [
     4.1,
     54.4,
     15.8,
     19
    ],
    [
     4.2,
     56.5,
     10.5,
     19
    ],
    [
     4.3,
     55.6,
     15.8,
     19
    ],
    [
     4.4,
     55.6,
     10.5,
     19
    ],
    [
     4.5,
     57.3,
     5.3,
     19
    ],
    [
     4.61,
     57.7,
     5.3,
     19
    ],
    [
     4.71,
     57.5,
     5.3,
     19
    ],
    [
     4.8,
     57.2,
     5.3,
     19
    ],
    [
     4.91,
     56.7,
     5.3,
     19
    ],
    [
     5.0,
     54.8,
     10.5,
     19
    ],
    [
     5.11,
     56.3,
     5.3,
     19
    ],
    [
     5.21,
     56.1,
     5.3,
     19
    ],
    [
     5.3,
     55.8,
     5.3,
     19
    ],
    [
     5.41,
     53.9,
     10.5,
     19
    ],
    [
     5.51,
     56.0,
     5.3,
     19
    ],
    [
     5.61,
     56.3,
     10.5,
     19
    ],
    [
     5.71,
     56.3,
     10.5,
     19
    ],
    [
     5.81,
     56.1,
     10.0,
     20
    ],
    [
     5.91,
     56.1,
     10.5,
     19
    ],
    [
     6.01,
     56.3,
     10.5,
     19
    ],
    [
     6.11,
     56.4,
     10.5,
     19
    ],
    [
     6.21,
     55.6,
     10.5,
     19
    ],
    [
     6.31,
     57.1,
     10.5,
     19
    ],
    [
     6.41,
     57.4,
     0.0,
     20
    ],
    [
     6.51,
     58.2,
     0.0,
     20
    ],
    [
     6.61,
     57.2,
     5.3,
     19
    ],
    [
     6.71,
     59.3,
     0.0,
     19
    ],
    [
     6.81,
     56.4,
     10.5,
     19
    ],
    [
     6.91,
     58.7,
     0.0,
     19
    ],
    [
     7.01,
     58.6,
     0.0,
     19
    ],
    [
     7.11,
     56.9,
     5.3,
     19
    ],
    [
     7.21,
     57.8,
     0.0,
     20
    ],
    [
     7.31,
     56.1,
     10.0,
     20
    ],
    [
     7.41,
     56.6,
     10.0,
     20
    ],
    [
     7.51,
     56.6,
     5.0,
     20
    ],
    [
     7.61,
     56.3,
     5.0,
     20
    ],
    [
     7.71,
     56.4,
     5.0,
     20
    ],
    [
     7.81,
     55.0,
     10.0,
     20
    ],
    [
     7.91,
     56.1,
     5.0,
     20
    ],
    [
     8.01,
     55.9,
     10.0,
     20
    ],
    [
     8.11,
     55.8,
     10.0,
     20
    ],
    [
     8.21,
     55.6,
     10.0,
     20
    ],
    [
     8.31,
     55.2,
     10.0,
     20
    ],
    [
     8.41,
     53.8,
     15.0,
     20
    ],
    [
     8.51,
     53.7,
     15.0,
     20
    ],
    [
     8.61,
     53.2,
     15.0,
     20
    ],
    [
     8.71,
     52.5,
     15.0,
     20
    ],
    [
     8.81,
     52.0,
     15.8,
     19
    ],
    [
     8.91,
     50.7,
     25.0,
     20
    ],
    [
     9.01,
     50.5,
     30.0,
     20
    ],
    [
     9.11,
     50.5,
     30.0,
     20
    ],
    [
     9.21,
     50.3,
     30.0,
     20
    ],
    [
     9.31,
     49.6,
     30.0,
     20
    ],
    [
     9.41,
     49.1,
     30.0,
     20
    ],
    [
     9.51,
     49.0,
     30.0,
     20
    ],
    [
     9.61,
     48.3,
     30.0,
     20
    ],
    [
     9.71,
     46.7,
     45.0,
     20
    ],
    [
     9.81,
     46.5,
     50.0,
     20
    ],
    [
     9.91,
     46.8,
     45.0,
     20
    ]
   ],
   "students": [
    {
     "id": 23,
     "n": 11,
     "mean": 30.0,
     "low_pct": 100.0,
     "cue": "looking right",
     "alerts": 0,
     "daisee": 67.4,
     "daisee_aux": [
      2.19,
      0.68,
      0.33,
      0.34
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 7,
     "n": 82,
     "mean": 46.5,
     "low_pct": 89.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 100.0,
     "daisee_aux": [
      3.16,
      0.26,
      0.53,
      -0.21
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 4,
     "n": 83,
     "mean": 49.1,
     "low_pct": 41.0,
     "cue": "glancing left",
     "alerts": 1,
     "daisee": 100.0,
     "daisee_aux": [
      3.04,
      -0.64,
      0.27,
      -0.19
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 10,
     "n": 83,
     "mean": 49.3,
     "low_pct": 45.8,
     "cue": "hand raised",
     "alerts": 0,
     "daisee": 96.9,
     "daisee_aux": [
      2.76,
      0.45,
      0.3,
      -0.05
     ],
     "scb": "reading",
     "scb_p": 0.24
    },
    {
     "id": 15,
     "n": 71,
     "mean": 50.1,
     "low_pct": 47.9,
     "cue": "glancing left",
     "alerts": 0,
     "daisee": 99.2,
     "daisee_aux": [
      3.05,
      -0.1,
      0.35,
      -0.13
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 18,
     "n": 83,
     "mean": 50.4,
     "low_pct": 38.6,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 74.7,
     "daisee_aux": [
      2.34,
      1.16,
      0.2,
      0.01
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 6,
     "n": 83,
     "mean": 52.4,
     "low_pct": 22.9,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 81.5,
     "daisee_aux": [
      2.46,
      0.65,
      0.15,
      0.01
     ],
     "scb": "reading",
     "scb_p": 0.32
    },
    {
     "id": 22,
     "n": 28,
     "mean": 52.7,
     "low_pct": 7.1,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 100.0,
     "daisee_aux": [
      2.95,
      0.68,
      0.46,
      -0.03
     ],
     "scb": "reading",
     "scb_p": 0.52
    },
    {
     "id": 12,
     "n": 82,
     "mean": 53.2,
     "low_pct": 12.2,
     "cue": "reading",
     "alerts": 0,
     "daisee": 96.5,
     "daisee_aux": [
      2.73,
      0.99,
      0.34,
      0.02
     ],
     "scb": "reading",
     "scb_p": 0.59
    },
    {
     "id": 8,
     "n": 83,
     "mean": 53.4,
     "low_pct": 24.1,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 68.2,
     "daisee_aux": [
      2.29,
      0.87,
      0.11,
      0.06
     ],
     "scb": "reading",
     "scb_p": 0.27
    },
    {
     "id": 1,
     "n": 82,
     "mean": 53.7,
     "low_pct": 25.6,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 100.0,
     "daisee_aux": [
      3.17,
      0.15,
      0.5,
      -0.21
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 3,
     "n": 21,
     "mean": 54.2,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 81.6,
     "daisee_aux": [
      2.39,
      0.6,
      0.13,
      0.04
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 17,
     "n": 82,
     "mean": 54.2,
     "low_pct": 13.4,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 93.1,
     "daisee_aux": [
      2.53,
      0.89,
      0.3,
      -0.01
     ],
     "scb": "reading",
     "scb_p": 0.34
    },
    {
     "id": 11,
     "n": 83,
     "mean": 55.0,
     "low_pct": 16.9,
     "cue": "glancing left",
     "alerts": 0,
     "daisee": 100.0,
     "daisee_aux": [
      3.1,
      -0.51,
      0.32,
      -0.25
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 14,
     "n": 83,
     "mean": 55.4,
     "low_pct": 7.2,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 95.1,
     "daisee_aux": [
      2.95,
      0.73,
      0.5,
      -0.12
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 20,
     "n": 82,
     "mean": 55.6,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 89.1,
     "daisee_aux": [
      2.81,
      0.55,
      0.37,
      -0.12
     ],
     "scb": "leaning on table",
     "scb_p": 0.18
    },
    {
     "id": 19,
     "n": 74,
     "mean": 55.9,
     "low_pct": 23.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 96.0,
     "daisee_aux": [
      2.63,
      0.4,
      0.29,
      -0.05
     ],
     "scb": "reading",
     "scb_p": 0.18
    },
    {
     "id": 13,
     "n": 83,
     "mean": 56.0,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 100.0,
     "daisee_aux": [
      3.13,
      0.41,
      0.56,
      -0.16
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 16,
     "n": 83,
     "mean": 56.4,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 96.0,
     "daisee_aux": [
      2.74,
      0.58,
      0.32,
      -0.14
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 9,
     "n": 82,
     "mean": 57.2,
     "low_pct": 12.2,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 100.0,
     "daisee_aux": [
      3.21,
      0.34,
      0.52,
      -0.23
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 5,
     "n": 82,
     "mean": 58.2,
     "low_pct": 0.0,
     "cue": "reading",
     "alerts": 0,
     "daisee": 92.5,
     "daisee_aux": [
      2.6,
      0.68,
      0.11,
      0.16
     ],
     "scb": "reading",
     "scb_p": 0.74
    },
    {
     "id": 2,
     "n": 82,
     "mean": 58.7,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 98.2,
     "daisee_aux": [
      2.62,
      0.62,
      0.3,
      -0.09
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    }
   ],
   "alerts": [
    {
     "t": 8.6,
     "student": 4,
     "kind": "sudden",
     "score": 39.8,
     "cue": "glancing left",
     "recommendation": "Direct an inclusive, low-stakes retrieval question toward that sector of the room, or use a vocal inflection change to refocus eyes on the visual aid.",
     "suggestion": "Wait one brief observation cycle; isolated look-aways are normal cognitive breathers."
    }
   ],
   "seats": [
    {
     "id": 13,
     "cx": 0.9303,
     "cy": 0.5561,
     "w": 0.0624,
     "h": 0.1219,
     "mean": 56.0,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "n": 83,
     "series": [
      [
       1.5,
       51.5
      ],
      [
       2.0,
       51.1
      ],
      [
       2.5,
       51.6
      ],
      [
       3.0,
       53.7
      ],
      [
       3.5,
       53.9
      ],
      [
       4.0,
       56.6
      ],
      [
       4.5,
       59.3
      ],
      [
       5.0,
       60.3
      ],
      [
       5.5,
       61.4
      ],
      [
       6.0,
       59.3
      ],
      [
       6.5,
       59.3
      ],
      [
       7.0,
       56.9
      ],
      [
       7.5,
       55.2
      ],
      [
       8.0,
       55.2
      ],
      [
       8.5,
       55.5
      ],
      [
       9.0,
       55.3
      ],
      [
       9.5,
       54.2
      ],
      [
       10.0,
       53.0
      ]
     ]
    },
    {
     "id": 11,
     "cx": 0.2844,
     "cy": 0.6319,
     "w": 0.0673,
     "h": 0.1697,
     "mean": 55.0,
     "low_pct": 16.9,
     "cue": "glancing left",
     "n": 83,
     "series": [
      [
       1.5,
       49.8
      ],
      [
       2.0,
       49.7
      ],
      [
       2.5,
       49.3
      ],
      [
       3.0,
       50.1
      ],
      [
       3.5,
       53.0
      ],
      [
       4.0,
       57.1
      ],
      [
       4.5,
       58.5
      ],
      [
       5.0,
       55.4
      ],
      [
       5.5,
       53.6
      ],
      [
       6.0,
       56.2
      ],
      [
       6.5,
       59.7
      ],
      [
       7.0,
       60.2
      ],
      [
       7.5,
       57.1
      ],
      [
       8.0,
       56.4
      ],
      [
       8.5,
       56.5
      ],
      [
       9.0,
       55.9
      ],
      [
       9.5,
       53.3
      ],
      [
       10.0,
       51.5
      ]
     ]
    },
    {
     "id": 6,
     "cx": 0.8038,
     "cy": 0.4137,
     "w": 0.0375,
     "h": 0.0982,
     "mean": 52.4,
     "low_pct": 22.9,
     "cue": "reading / taking notes",
     "n": 83,
     "series": [
      [
       1.5,
       44.2
      ],
      [
       2.0,
       44.3
      ],
      [
       2.5,
       47.2
      ],
      [
       3.0,
       49.8
      ],
      [
       3.5,
       50.9
      ],
      [
       4.0,
       53.2
      ],
      [
       4.5,
       54.6
      ],
      [
       5.0,
       54.3
      ],
      [
       5.5,
       55.3
      ],
      [
       6.0,
       55.8
      ],
      [
       6.5,
       57.5
      ],
      [
       7.0,
       58.0
      ],
      [
       7.5,
       55.9
      ],
      [
       8.0,
       55.4
      ],
      [
       8.5,
       52.9
      ],
      [
       9.0,
       49.6
      ],
      [
       9.5,
       46.5
      ],
      [
       10.0,
       47.1
      ]
     ]
    },
    {
     "id": 19,
     "cx": 0.4518,
     "cy": 0.3951,
     "w": 0.0428,
     "h": 0.1233,
     "mean": 55.9,
     "low_pct": 23.0,
     "cue": "reading / writing",
     "n": 74,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       47.9
      ],
      [
       2.5,
       47.5
      ],
      [
       3.5,
       48.8
      ],
      [
       4.0,
       49.0
      ],
      [
       4.5,
       51.5
      ],
      [
       5.0,
       56.3
      ],
      [
       5.5,
       57.7
      ],
      [
       6.0,
       58.7
      ],
      [
       6.5,
       60.2
      ],
      [
       7.0,
       61.8
      ],
      [
       7.5,
       61.0
      ],
      [
       8.0,
       61.5
      ],
      [
       8.5,
       59.7
      ],
      [
       9.0,
       57.7
      ],
      [
       9.5,
       55.1
      ],
      [
       10.0,
       53.2
      ]
     ]
    },
    {
     "id": 14,
     "cx": 0.815,
     "cy": 0.5791,
     "w": 0.0753,
     "h": 0.2436,
     "mean": 55.4,
     "low_pct": 7.2,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       1.5,
       51.2
      ],
      [
       2.0,
       50.9
      ],
      [
       2.5,
       51.6
      ],
      [
       3.0,
       52.6
      ],
      [
       3.5,
       52.6
      ],
      [
       4.0,
       50.9
      ],
      [
       4.5,
       59.8
      ],
      [
       5.0,
       56.7
      ],
      [
       5.5,
       56.6
      ],
      [
       6.0,
       62.0
      ],
      [
       6.5,
       55.6
      ],
      [
       7.0,
       55.4
      ],
      [
       7.5,
       60.6
      ],
      [
       8.0,
       60.4
      ],
      [
       8.5,
       58.3
      ],
      [
       9.0,
       55.6
      ],
      [
       9.5,
       48.9
      ],
      [
       10.0,
       51.9
      ]
     ]
    },
    {
     "id": 15,
     "cx": 0.6482,
     "cy": 0.5638,
     "w": 0.0428,
     "h": 0.1248,
     "mean": 50.1,
     "low_pct": 47.9,
     "cue": "glancing left",
     "n": 71,
     "series": [
      [
       1.5,
       46.7
      ],
      [
       2.0,
       46.2
      ],
      [
       2.5,
       47.2
      ],
      [
       3.0,
       48.4
      ],
      [
       3.5,
       44.5
      ],
      [
       4.0,
       50.3
      ],
      [
       4.5,
       51.5
      ],
      [
       5.0,
       52.6
      ],
      [
       5.5,
       50.6
      ],
      [
       6.0,
       45.2
      ],
      [
       6.5,
       51.1
      ],
      [
       7.0,
       44.6
      ],
      [
       7.5,
       58.5
      ],
      [
       8.0,
       53.3
      ],
      [
       8.5,
       57.3
      ]
     ]
    },
    {
     "id": 8,
     "cx": 0.8993,
     "cy": 0.4154,
     "w": 0.045,
     "h": 0.1103,
     "mean": 53.4,
     "low_pct": 24.1,
     "cue": "reading / taking notes",
     "n": 83,
     "series": [
      [
       1.5,
       48.2
      ],
      [
       2.0,
       47.2
      ],
      [
       2.5,
       47.3
      ],
      [
       3.0,
       48.7
      ],
      [
       3.5,
       49.3
      ],
      [
       4.0,
       52.6
      ],
      [
       4.5,
       55.7
      ],
      [
       5.0,
       56.2
      ],
      [
       5.5,
       57.6
      ],
      [
       6.0,
       56.3
      ],
      [
       6.5,
       57.5
      ],
      [
       7.0,
       57.4
      ],
      [
       7.5,
       56.0
      ],
      [
       8.0,
       54.9
      ],
      [
       8.5,
       53.7
      ],
      [
       9.0,
       53.1
      ],
      [
       9.5,
       52.5
      ],
      [
       10.0,
       51.9
      ]
     ]
    },
    {
     "id": 10,
     "cx": 0.9615,
     "cy": 0.3268,
     "w": 0.0413,
     "h": 0.0882,
     "mean": 49.3,
     "low_pct": 45.8,
     "cue": "hand raised",
     "n": 83,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       48.8
      ],
      [
       2.5,
       48.8
      ],
      [
       3.0,
       46.0
      ],
      [
       3.5,
       49.9
      ],
      [
       4.0,
       52.6
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       55.9
      ],
      [
       5.5,
       58.1
      ],
      [
       6.0,
       56.1
      ],
      [
       6.5,
       56.5
      ],
      [
       7.0,
       52.8
      ],
      [
       7.5,
       46.2
      ],
      [
       8.0,
       49.2
      ],
      [
       8.5,
       43.2
      ],
      [
       9.0,
       40.0
      ],
      [
       9.5,
       38.7
      ],
      [
       10.0,
       39.7
      ]
     ]
    },
    {
     "id": 18,
     "cx": 0.6139,
     "cy": 0.439,
     "w": 0.0387,
     "h": 0.1407,
     "mean": 50.4,
     "low_pct": 38.6,
     "cue": "reading / taking notes",
     "n": 83,
     "series": [
      [
       1.5,
       45.1
      ],
      [
       2.0,
       44.2
      ],
      [
       2.5,
       44.4
      ],
      [
       3.0,
       46.1
      ],
      [
       3.5,
       48.2
      ],
      [
       4.0,
       52.8
      ],
      [
       4.5,
       55.7
      ],
      [
       5.0,
       53.9
      ],
      [
       5.5,
       53.0
      ],
      [
       6.0,
       52.7
      ],
      [
       6.5,
       54.2
      ],
      [
       7.0,
       53.8
      ],
      [
       7.5,
       52.5
      ],
      [
       8.0,
       52.7
      ],
      [
       8.5,
       51.8
      ],
      [
       9.0,
       49.0
      ],
      [
       9.5,
       45.1
      ],
      [
       10.0,
       42.5
      ]
     ]
    },
    {
     "id": 16,
     "cx": 0.345,
     "cy": 0.4088,
     "w": 0.0323,
     "h": 0.1087,
     "mean": 56.4,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       1.5,
       53.8
      ],
      [
       2.0,
       53.3
      ],
      [
       2.5,
       52.5
      ],
      [
       3.0,
       53.2
      ],
      [
       3.5,
       56.1
      ],
      [
       4.0,
       60.3
      ],
      [
       4.5,
       62.0
      ],
      [
       5.0,
       57.9
      ],
      [
       5.5,
       55.4
      ],
      [
       6.0,
       57.2
      ],
      [
       6.5,
       60.3
      ],
      [
       7.0,
       62.2
      ],
      [
       7.5,
       59.1
      ],
      [
       8.0,
       55.9
      ],
      [
       8.5,
       53.9
      ],
      [
       9.0,
       52.8
      ],
      [
       9.5,
       52.3
      ],
      [
       10.0,
       52.4
      ]
     ]
    },
    {
     "id": 4,
     "cx": 0.0345,
     "cy": 0.4849,
     "w": 0.0342,
     "h": 0.0998,
     "mean": 49.1,
     "low_pct": 41.0,
     "cue": "glancing left",
     "n": 83,
     "series": [
      [
       1.5,
       49.4
      ],
      [
       2.0,
       48.2
      ],
      [
       2.5,
       47.5
      ],
      [
       3.0,
       45.4
      ],
      [
       3.5,
       48.4
      ],
      [
       4.0,
       56.2
      ],
      [
       4.5,
       51.5
      ],
      [
       5.0,
       53.7
      ],
      [
       5.5,
       53.2
      ],
      [
       6.0,
       57.1
      ],
      [
       6.5,
       60.6
      ],
      [
       7.0,
       60.5
      ],
      [
       7.5,
       55.3
      ],
      [
       8.0,
       54.4
      ],
      [
       8.5,
       36.6
      ],
      [
       9.0,
       31.6
      ],
      [
       9.5,
       32.0
      ],
      [
       10.0,
       31.0
      ]
     ]
    },
    {
     "id": 1,
     "cx": 0.518,
     "cy": 0.6667,
     "w": 0.0884,
     "h": 0.2396,
     "mean": 53.7,
     "low_pct": 25.6,
     "cue": "reading / taking notes",
     "n": 82,
     "series": [
      [
       2.0,
       48.0
      ],
      [
       2.5,
       48.1
      ],
      [
       3.0,
       48.5
      ],
      [
       3.5,
       49.5
      ],
      [
       4.0,
       53.5
      ],
      [
       4.5,
       56.6
      ],
      [
       5.0,
       56.3
      ],
      [
       5.5,
       53.9
      ],
      [
       6.0,
       54.7
      ],
      [
       6.5,
       57.8
      ],
      [
       7.0,
       59.4
      ],
      [
       7.5,
       57.5
      ],
      [
       8.0,
       57.1
      ],
      [
       8.5,
       55.6
      ],
      [
       9.0,
       53.4
      ],
      [
       9.5,
       50.9
      ],
      [
       10.0,
       49.0
      ]
     ]
    },
    {
     "id": 2,
     "cx": 0.9286,
     "cy": 0.7402,
     "w": 0.1212,
     "h": 0.2236,
     "mean": 58.7,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "n": 82,
     "series": [
      [
       2.0,
       54.6
      ],
      [
       2.5,
       55.0
      ],
      [
       3.0,
       56.3
      ],
      [
       3.5,
       56.0
      ],
      [
       4.0,
       59.1
      ],
      [
       4.5,
       62.1
      ],
      [
       5.0,
       63.0
      ],
      [
       5.5,
       63.4
      ],
      [
       6.0,
       61.5
      ],
      [
       6.5,
       61.7
      ],
      [
       7.0,
       61.5
      ],
      [
       7.5,
       59.6
      ],
      [
       8.0,
       58.8
      ],
      [
       8.5,
       57.1
      ],
      [
       9.0,
       56.1
      ],
      [
       9.5,
       54.4
      ],
      [
       10.0,
       54.8
      ]
     ]
    },
    {
     "id": 3,
     "cx": 0.4372,
     "cy": 0.437,
     "w": 0.0429,
     "h": 0.1122,
     "mean": 54.2,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "n": 21,
     "series": [
      [
       2.0,
       53.5
      ],
      [
       2.5,
       53.4
      ],
      [
       3.0,
       54.9
      ],
      [
       3.5,
       56.3
      ],
      [
       6.0,
       53.3
      ],
      [
       6.5,
       53.8
      ]
     ]
    },
    {
     "id": 7,
     "cx": 0.0882,
     "cy": 0.4951,
     "w": 0.0601,
     "h": 0.1699,
     "mean": 46.5,
     "low_pct": 89.0,
     "cue": "reading / taking notes",
     "n": 82,
     "series": [
      [
       2.0,
       42.2
      ],
      [
       2.5,
       42.2
      ],
      [
       3.0,
       43.1
      ],
      [
       3.5,
       45.8
      ],
      [
       4.0,
       48.4
      ],
      [
       4.5,
       48.9
      ],
      [
       5.0,
       46.2
      ],
      [
       5.5,
       44.9
      ],
      [
       6.0,
       47.7
      ],
      [
       6.5,
       50.9
      ],
      [
       7.0,
       51.6
      ],
      [
       7.5,
       48.5
      ],
      [
       8.0,
       48.1
      ],
      [
       8.5,
       47.8
      ],
      [
       9.0,
       46.7
      ],
      [
       9.5,
       43.5
      ],
      [
       10.0,
       41.5
      ]
     ]
    },
    {
     "id": 5,
     "cx": 0.5216,
     "cy": 0.3936,
     "w": 0.0414,
     "h": 0.106,
     "mean": 58.2,
     "low_pct": 0.0,
     "cue": "reading",
     "n": 82,
     "series": [
      [
       2.0,
       52.9
      ],
      [
       2.5,
       53.3
      ],
      [
       3.0,
       54.9
      ],
      [
       3.5,
       57.1
      ],
      [
       4.0,
       60.7
      ],
      [
       4.5,
       62.6
      ],
      [
       5.0,
       59.5
      ],
      [
       5.5,
       58.1
      ],
      [
       6.0,
       59.0
      ],
      [
       6.5,
       61.9
      ],
      [
       7.0,
       63.3
      ],
      [
       7.5,
       60.9
      ],
      [
       8.0,
       60.1
      ],
      [
       8.5,
       58.4
      ],
      [
       9.0,
       56.6
      ],
      [
       9.5,
       54.2
      ],
      [
       10.0,
       52.7
      ]
     ]
    },
    {
     "id": 9,
     "cx": 0.7285,
     "cy": 0.7596,
     "w": 0.0964,
     "h": 0.2611,
     "mean": 57.2,
     "low_pct": 12.2,
     "cue": "reading / taking notes",
     "n": 82,
     "series": [
      [
       2.0,
       49.4
      ],
      [
       2.5,
       49.8
      ],
      [
       3.0,
       52.0
      ],
      [
       3.5,
       54.0
      ],
      [
       4.0,
       60.2
      ],
      [
       4.5,
       64.1
      ],
      [
       5.0,
       62.7
      ],
      [
       5.5,
       61.0
      ],
      [
       6.0,
       60.9
      ],
      [
       6.5,
       62.6
      ],
      [
       7.0,
       61.5
      ],
      [
       7.5,
       59.1
      ],
      [
       8.0,
       58.0
      ],
      [
       8.5,
       56.5
      ],
      [
       9.0,
       54.1
      ],
      [
       9.5,
       51.7
      ],
      [
       10.0,
       50.1
      ]
     ]
    },
    {
     "id": 20,
     "cx": 0.6946,
     "cy": 0.4617,
     "w": 0.0478,
     "h": 0.1422,
     "mean": 55.6,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "n": 82,
     "series": [
      [
       2.0,
       51.8
      ],
      [
       2.5,
       52.5
      ],
      [
       3.0,
       54.2
      ],
      [
       3.5,
       55.1
      ],
      [
       4.0,
       58.0
      ],
      [
       4.5,
       58.5
      ],
      [
       5.0,
       56.1
      ],
      [
       5.5,
       54.9
      ],
      [
       6.0,
       55.4
      ],
      [
       6.5,
       58.1
      ],
      [
       7.0,
       59.5
      ],
      [
       7.5,
       58.4
      ],
      [
       8.0,
       57.4
      ],
      [
       8.5,
       55.8
      ],
      [
       9.0,
       54.0
      ],
      [
       9.5,
       52.4
      ],
      [
       10.0,
       51.0
      ]
     ]
    },
    {
     "id": 12,
     "cx": 0.5225,
     "cy": 0.4936,
     "w": 0.0435,
     "h": 0.0841,
     "mean": 53.2,
     "low_pct": 12.2,
     "cue": "reading",
     "n": 82,
     "series": [
      [
       2.0,
       47.1
      ],
      [
       2.5,
       47.8
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       52.0
      ],
      [
       4.0,
       55.1
      ],
      [
       4.5,
       56.3
      ],
      [
       5.0,
       54.2
      ],
      [
       5.5,
       52.5
      ],
      [
       6.0,
       53.5
      ],
      [
       6.5,
       56.6
      ],
      [
       7.0,
       58.5
      ],
      [
       7.5,
       55.6
      ],
      [
       8.0,
       55.0
      ],
      [
       8.5,
       54.2
      ],
      [
       9.0,
       53.4
      ],
      [
       9.5,
       51.2
      ],
      [
       10.0,
       49.9
      ]
     ]
    },
    {
     "id": 17,
     "cx": 0.7205,
     "cy": 0.4155,
     "w": 0.0274,
     "h": 0.1054,
     "mean": 54.2,
     "low_pct": 13.4,
     "cue": "reading / taking notes",
     "n": 82,
     "series": [
      [
       2.0,
       48.8
      ],
      [
       2.5,
       49.2
      ],
      [
       3.0,
       50.2
      ],
      [
       3.5,
       52.0
      ],
      [
       4.0,
       56.6
      ],
      [
       4.5,
       58.7
      ],
      [
       5.0,
       56.6
      ],
      [
       5.5,
       56.4
      ],
      [
       6.0,
       56.5
      ],
      [
       6.5,
       58.5
      ],
      [
       7.0,
       57.8
      ],
      [
       7.5,
       56.0
      ],
      [
       8.0,
       55.3
      ],
      [
       8.5,
       54.1
      ],
      [
       9.0,
       52.1
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       47.9
      ]
     ]
    },
    {
     "id": 22,
     "cx": 0.8559,
     "cy": 0.4524,
     "w": 0.033,
     "h": 0.0651,
     "mean": 52.7,
     "low_pct": 7.1,
     "cue": "reading / taking notes",
     "n": 28,
     "series": [
      [
       7.0,
       54.2
      ],
      [
       7.5,
       54.6
      ],
      [
       8.0,
       54.3
      ],
      [
       8.5,
       54.8
      ],
      [
       9.0,
       54.2
      ],
      [
       9.5,
       53.9
      ],
      [
       10.0,
       32.0
      ]
     ]
    },
    {
     "id": 23,
     "cx": 0.7435,
     "cy": 0.5057,
     "w": 0.0296,
     "h": 0.0806,
     "mean": 30.0,
     "low_pct": 100.0,
     "cue": "looking right",
     "n": 11,
     "series": [
      [
       9.0,
       30.0
      ],
      [
       9.5,
       30.0
      ],
      [
       10.0,
       30.0
      ]
     ]
    }
   ],
   "class_mean": 53.56,
   "low_share": 21.9
  },
  {
   "key": "room_B_11",
   "title": "Seminar room B (video 11)",
   "room": "room_B",
   "labelled": true,
   "video": "media/room_B_11.mp4",
   "poster": "media/room_B_11.jpg",
   "frame": [
    1920,
    1080
   ],
   "duration": 10.8,
   "students_tracked": 14,
   "timeline": [
    [
     1.8,
     48.7,
     38.5,
     13
    ],
    [
     1.9,
     47.7,
     50.0,
     14
    ],
    [
     2.0,
     47.4,
     50.0,
     14
    ],
    [
     2.1,
     47.4,
     50.0,
     14
    ],
    [
     2.2,
     47.4,
     50.0,
     14
    ],
    [
     2.3,
     47.3,
     50.0,
     14
    ],
    [
     2.4,
     47.2,
     50.0,
     14
    ],
    [
     2.5,
     47.3,
     50.0,
     14
    ],
    [
     2.6,
     47.4,
     50.0,
     14
    ],
    [
     2.7,
     47.4,
     64.3,
     14
    ],
    [
     2.8,
     47.4,
     64.3,
     14
    ],
    [
     2.9,
     47.4,
     64.3,
     14
    ],
    [
     3.0,
     47.5,
     64.3,
     14
    ],
    [
     3.1,
     47.8,
     57.1,
     14
    ],
    [
     3.2,
     48.0,
     42.9,
     14
    ],
    [
     3.3,
     48.3,
     42.9,
     14
    ],
    [
     3.4,
     48.8,
     42.9,
     14
    ],
    [
     3.5,
     48.9,
     42.9,
     14
    ],
    [
     3.6,
     48.8,
     42.9,
     14
    ],
    [
     3.7,
     48.8,
     42.9,
     14
    ],
    [
     3.8,
     49.4,
     35.7,
     14
    ],
    [
     3.9,
     49.5,
     35.7,
     14
    ],
    [
     4.0,
     49.5,
     35.7,
     14
    ],
    [
     4.1,
     49.1,
     35.7,
     14
    ],
    [
     4.2,
     48.5,
     50.0,
     14
    ],
    [
     4.3,
     48.8,
     50.0,
     14
    ],
    [
     4.4,
     48.4,
     50.0,
     14
    ],
    [
     4.5,
     48.6,
     50.0,
     14
    ],
    [
     4.6,
     48.9,
     42.9,
     14
    ],
    [
     4.7,
     48.9,
     42.9,
     14
    ],
    [
     4.8,
     49.2,
     35.7,
     14
    ],
    [
     4.9,
     49.1,
     42.9,
     14
    ],
    [
     5.0,
     48.9,
     42.9,
     14
    ],
    [
     5.1,
     48.7,
     42.9,
     14
    ],
    [
     5.2,
     48.9,
     42.9,
     14
    ],
    [
     5.3,
     48.9,
     42.9,
     14
    ],
    [
     5.4,
     49.0,
     35.7,
     14
    ],
    [
     5.5,
     49.3,
     35.7,
     14
    ],
    [
     5.6,
     49.6,
     35.7,
     14
    ],
    [
     5.7,
     49.9,
     35.7,
     14
    ],
    [
     5.8,
     50.6,
     28.6,
     14
    ],
    [
     5.9,
     51.2,
     28.6,
     14
    ],
    [
     6.0,
     50.9,
     28.6,
     14
    ],
    [
     6.1,
     51.8,
     28.6,
     14
    ],
    [
     6.2,
     52.0,
     28.6,
     14
    ],
    [
     6.3,
     52.3,
     28.6,
     14
    ],
    [
     6.4,
     52.8,
     14.3,
     14
    ],
    [
     6.5,
     51.9,
     14.3,
     14
    ],
    [
     6.6,
     52.6,
     14.3,
     14
    ],
    [
     6.7,
     52.1,
     14.3,
     14
    ],
    [
     6.8,
     52.9,
     14.3,
     14
    ],
    [
     6.9,
     53.2,
     14.3,
     14
    ],
    [
     7.0,
     53.2,
     14.3,
     14
    ],
    [
     7.1,
     52.4,
     14.3,
     14
    ],
    [
     7.2,
     52.4,
     14.3,
     14
    ],
    [
     7.3,
     53.9,
     14.3,
     14
    ],
    [
     7.4,
     53.7,
     14.3,
     14
    ],
    [
     7.5,
     52.5,
     14.3,
     14
    ],
    [
     7.6,
     52.3,
     14.3,
     14
    ],
    [
     7.7,
     52.4,
     14.3,
     14
    ],
    [
     7.8,
     52.8,
     14.3,
     14
    ],
    [
     7.9,
     52.3,
     21.4,
     14
    ],
    [
     8.0,
     52.1,
     21.4,
     14
    ],
    [
     8.1,
     51.8,
     28.6,
     14
    ],
    [
     8.2,
     51.6,
     28.6,
     14
    ],
    [
     8.3,
     51.1,
     35.7,
     14
    ],
    [
     8.4,
     50.7,
     42.9,
     14
    ],
    [
     8.5,
     50.4,
     42.9,
     14
    ],
    [
     8.6,
     50.0,
     50.0,
     14
    ],
    [
     8.7,
     49.2,
     50.0,
     14
    ],
    [
     8.8,
     48.6,
     57.1,
     14
    ],
    [
     8.9,
     47.9,
     64.3,
     14
    ],
    [
     9.0,
     47.1,
     64.3,
     14
    ],
    [
     9.1,
     45.7,
     64.3,
     14
    ],
    [
     9.2,
     46.1,
     69.2,
     13
    ],
    [
     9.3,
     45.0,
     76.9,
     13
    ],
    [
     9.4,
     45.3,
     76.9,
     13
    ],
    [
     9.5,
     45.2,
     76.9,
     13
    ],
    [
     9.6,
     45.1,
     69.2,
     13
    ],
    [
     9.7,
     44.9,
     71.4,
     14
    ],
    [
     9.8,
     44.0,
     71.4,
     14
    ],
    [
     9.9,
     43.6,
     78.6,
     14
    ],
    [
     10.0,
     43.3,
     78.6,
     14
    ],
    [
     10.1,
     43.0,
     78.6,
     14
    ],
    [
     10.2,
     44.1,
     76.9,
     13
    ],
    [
     10.3,
     44.1,
     76.9,
     13
    ],
    [
     10.4,
     43.9,
     76.9,
     13
    ],
    [
     10.5,
     43.8,
     78.6,
     14
    ],
    [
     10.6,
     44.0,
     78.6,
     14
    ],
    [
     10.7,
     44.3,
     78.6,
     14
    ],
    [
     10.8,
     44.6,
     78.6,
     14
    ]
   ],
   "students": [
    {
     "id": 5,
     "n": 91,
     "mean": 37.2,
     "low_pct": 100.0,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 68.0,
     "daisee_aux": [
      2.33,
      0.64,
      0.24,
      0.29
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 1,
     "n": 90,
     "mean": 43.5,
     "low_pct": 73.3,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 97.2,
     "daisee_aux": [
      2.98,
      0.58,
      0.49,
      -0.05
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 4,
     "n": 91,
     "mean": 43.7,
     "low_pct": 93.4,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.4,
     "daisee_aux": [
      2.31,
      0.77,
      0.09,
      0.24
     ],
     "scb": "reading",
     "scb_p": 0.19
    },
    {
     "id": 9,
     "n": 91,
     "mean": 44.0,
     "low_pct": 81.3,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.8,
     "daisee_aux": [
      2.36,
      0.7,
      0.2,
      0.29
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 6,
     "n": 91,
     "mean": 48.0,
     "low_pct": 29.7,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.4,
     "daisee_aux": [
      2.33,
      0.85,
      0.09,
      0.26
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 14,
     "n": 88,
     "mean": 48.3,
     "low_pct": 54.5,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 66.6,
     "daisee_aux": [
      2.3,
      0.9,
      0.1,
      0.23
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 7,
     "n": 91,
     "mean": 49.1,
     "low_pct": 53.8,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 66.9,
     "daisee_aux": [
      2.43,
      0.78,
      0.11,
      0.34
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 11,
     "n": 86,
     "mean": 49.4,
     "low_pct": 23.3,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 68.0,
     "daisee_aux": [
      2.36,
      0.68,
      0.19,
      0.29
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 10,
     "n": 91,
     "mean": 49.9,
     "low_pct": 39.6,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.7,
     "daisee_aux": [
      2.43,
      0.72,
      0.12,
      0.35
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 2,
     "n": 91,
     "mean": 50.7,
     "low_pct": 38.5,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.8,
     "daisee_aux": [
      2.4,
      0.74,
      0.1,
      0.27
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 12,
     "n": 91,
     "mean": 52.5,
     "low_pct": 22.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.8,
     "daisee_aux": [
      2.42,
      0.73,
      0.11,
      0.31
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 8,
     "n": 91,
     "mean": 54.1,
     "low_pct": 15.4,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.6,
     "daisee_aux": [
      2.44,
      0.74,
      0.09,
      0.31
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 3,
     "n": 91,
     "mean": 56.1,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 67.6,
     "daisee_aux": [
      2.42,
      0.85,
      0.25,
      0.27
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 13,
     "n": 91,
     "mean": 56.9,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 68.2,
     "daisee_aux": [
      2.38,
      0.74,
      0.17,
      0.31
     ],
     "scb": "hand-raising",
     "scb_p": 0.19
    }
   ],
   "alerts": [
    {
     "t": 5.4,
     "student": 1,
     "kind": "sustained",
     "score": 38.5,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 10.5,
     "student": 5,
     "kind": "sustained",
     "score": 29.9,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    }
   ],
   "seats": [
    {
     "id": 2,
     "cx": 0.5811,
     "cy": 0.5525,
     "w": 0.0687,
     "h": 0.1857,
     "mean": 50.7,
     "low_pct": 38.5,
     "cue": "reading / writing",
     "n": 91,
     "series": [
      [
       2.0,
       49.9
      ],
      [
       2.5,
       49.5
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.4
      ],
      [
       4.0,
       50.2
      ],
      [
       4.5,
       50.9
      ],
      [
       5.0,
       52.9
      ],
      [
       5.5,
       54.5
      ],
      [
       6.0,
       56.0
      ],
      [
       6.5,
       56.7
      ],
      [
       7.0,
       56.0
      ],
      [
       7.5,
       56.3
      ],
      [
       8.0,
       55.6
      ],
      [
       8.5,
       50.6
      ],
      [
       9.0,
       45.4
      ],
      [
       9.5,
       41.4
      ],
      [
       10.0,
       42.4
      ],
      [
       10.5,
       44.8
      ],
      [
       11.0,
       48.2
      ]
     ]
    },
    {
     "id": 4,
     "cx": 0.701,
     "cy": 0.4952,
     "w": 0.0533,
     "h": 0.143,
     "mean": 43.7,
     "low_pct": 93.4,
     "cue": "reading / writing",
     "n": 91,
     "series": [
      [
       2.0,
       41.9
      ],
      [
       2.5,
       41.0
      ],
      [
       3.0,
       43.3
      ],
      [
       3.5,
       43.7
      ],
      [
       4.0,
       48.7
      ],
      [
       4.5,
       44.9
      ],
      [
       5.0,
       43.1
      ],
      [
       5.5,
       38.5
      ],
      [
       6.0,
       36.8
      ],
      [
       6.5,
       36.8
      ],
      [
       7.0,
       39.9
      ],
      [
       7.5,
       42.4
      ],
      [
       8.0,
       44.0
      ],
      [
       8.5,
       45.7
      ],
      [
       9.0,
       47.5
      ],
      [
       9.5,
       48.5
      ],
      [
       10.0,
       49.0
      ],
      [
       10.5,
       49.1
      ],
      [
       11.0,
       49.8
      ]
     ]
    },
    {
     "id": 3,
     "cx": 0.9249,
     "cy": 0.624,
     "w": 0.0829,
     "h": 0.2227,
     "mean": 56.1,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "n": 91,
     "series": [
      [
       2.0,
       56.1
      ],
      [
       2.5,
       55.6
      ],
      [
       3.0,
       54.8
      ],
      [
       3.5,
       55.4
      ],
      [
       4.0,
       55.1
      ],
      [
       4.5,
       54.1
      ],
      [
       5.0,
       55.5
      ],
      [
       5.5,
       56.8
      ],
      [
       6.0,
       60.2
      ],
      [
       6.5,
       60.6
      ],
      [
       7.0,
       60.4
      ],
      [
       7.5,
       59.9
      ],
      [
       8.0,
       58.4
      ],
      [
       8.5,
       57.0
      ],
      [
       9.0,
       55.4
      ],
      [
       9.5,
       53.7
      ],
      [
       10.0,
       51.7
      ],
      [
       10.5,
       50.4
      ],
      [
       11.0,
       50.5
      ]
     ]
    },
    {
     "id": 5,
     "cx": 0.5249,
     "cy": 0.4727,
     "w": 0.0259,
     "h": 0.0789,
     "mean": 37.2,
     "low_pct": 100.0,
     "cue": "reading / writing",
     "n": 91,
     "series": [
      [
       2.0,
       37.4
      ],
      [
       2.5,
       37.6
      ],
      [
       3.0,
       40.0
      ],
      [
       3.5,
       43.2
      ],
      [
       4.0,
       43.3
      ],
      [
       4.5,
       41.8
      ],
      [
       5.0,
       40.4
      ],
      [
       5.5,
       39.2
      ],
      [
       6.0,
       37.5
      ],
      [
       6.5,
       36.3
      ],
      [
       7.0,
       37.0
      ],
      [
       7.5,
       36.1
      ],
      [
       8.0,
       40.7
      ],
      [
       8.5,
       36.6
      ],
      [
       9.0,
       32.2
      ],
      [
       9.5,
       30.6
      ],
      [
       10.0,
       29.7
      ],
      [
       10.5,
       30.4
      ],
      [
       11.0,
       30.0
      ]
     ]
    },
    {
     "id": 8,
     "cx": 0.8202,
     "cy": 0.5577,
     "w": 0.065,
     "h": 0.172,
     "mean": 54.1,
     "low_pct": 15.4,
     "cue": "reading / writing",
     "n": 91,
     "series": [
      [
       2.0,
       42.6
      ],
      [
       2.5,
       45.8
      ],
      [
       3.0,
       49.4
      ],
      [
       3.5,
       53.7
      ],
      [
       4.0,
       55.2
      ],
      [
       4.5,
       54.8
      ],
      [
       5.0,
       55.9
      ],
      [
       5.5,
       57.0
      ],
      [
       6.0,
       59.1
      ],
      [
       6.5,
       59.3
      ],
      [
       7.0,
       56.8
      ],
      [
       7.5,
       56.3
      ],
      [
       8.0,
       54.7
      ],
      [
       8.5,
       54.8
      ],
      [
       9.0,
       55.0
      ],
      [
       9.5,
       55.0
      ],
      [
       10.0,
       55.0
      ],
      [
       10.5,
       53.9
      ],
      [
       11.0,
       54.2
      ]
     ]
    },
    {
     "id": 10,
     "cx": 0.7525,
     "cy": 0.495,
     "w": 0.0538,
     "h": 0.1437,
     "mean": 49.9,
     "low_pct": 39.6,
     "cue": "reading / writing",
     "n": 91,
     "series": [
      [
       2.0,
       52.2
      ],
      [
       2.5,
       50.1
      ],
      [
       3.0,
       46.5
      ],
      [
       3.5,
       45.8
      ],
      [
       4.0,
       46.8
      ],
      [
       4.5,
       49.5
      ],
      [
       5.0,
       51.5
      ],
      [
       5.5,
       51.5
      ],
      [
       6.0,
       52.7
      ],
      [
       6.5,
       53.6
      ],
      [
       7.0,
       54.2
      ],
      [
       7.5,
       54.8
      ],
      [
       8.0,
       54.0
      ],
      [
       8.5,
       52.7
      ],
      [
       9.0,
       50.9
      ],
      [
       9.5,
       46.5
      ],
      [
       10.0,
       44.0
      ],
      [
       10.5,
       43.0
      ],
      [
       11.0,
       44.1
      ]
     ]
    },
    {
     "id": 6,
     "cx": 0.3293,
     "cy": 0.4912,
     "w": 0.0374,
     "h": 0.1264,
     "mean": 48.0,
     "low_pct": 29.7,
     "cue": "reading / writing",
     "n": 91,
     "series": [
      [
       2.0,
       51.7
      ],
      [
       2.5,
       51.7
      ],
      [
       3.0,
       51.3
      ],
      [
       3.5,
       51.4
      ],
      [
       4.0,
       50.6
      ],
      [
       4.5,
       50.3
      ],
      [
       5.0,
       50.5
      ],
      [
       5.5,
       51.1
      ],
      [
       6.0,
       52.6
      ],
      [
       6.5,
       52.6
      ],
      [
       7.0,
       52.6
      ],
      [
       7.5,
       52.3
      ],
      [
       8.0,
       51.3
      ],
      [
       8.5,
       48.2
      ],
      [
       9.0,
       42.9
      ],
      [
       9.5,
       36.4
      ],
      [
       10.0,
       35.3
      ],
      [
       10.5,
       33.6
      ],
      [
       11.0,
       35.6
      ]
     ]
    },
    {
     "id": 9,
     "cx": 0.5488,
     "cy": 0.4909,
     "w": 0.0509,
     "h": 0.1449,
     "mean": 44.0,
     "low_pct": 81.3,
     "cue": "reading / writing",
     "n": 91,
     "series": [
      [
       2.0,
       39.5
      ],
      [
       2.5,
       40.5
      ],
      [
       3.0,
       43.1
      ],
      [
       3.5,
       43.9
      ],
      [
       4.0,
       42.3
      ],
      [
       4.5,
       42.2
      ],
      [
       5.0,
       43.4
      ],
      [
       5.5,
       45.3
      ],
      [
       6.0,
       48.3
      ],
      [
       6.5,
       50.5
      ],
      [
       7.0,
       51.0
      ],
      [
       7.5,
       51.4
      ],
      [
       8.0,
       50.1
      ],
      [
       8.5,
       46.5
      ],
      [
       9.0,
       41.1
      ],
      [
       9.5,
       37.4
      ],
      [
       10.0,
       37.3
      ],
      [
       10.5,
       39.1
      ],
      [
       11.0,
       41.5
      ]
     ]
    },
    {
     "id": 14,
     "cx": 0.2766,
     "cy": 0.5376,
     "w": 0.0513,
     "h": 0.1627,
     "mean": 48.3,
     "low_pct": 54.5,
     "cue": "reading / writing",
     "n": 88,
     "series": [
      [
       2.0,
       51.7
      ],
      [
       2.5,
       50.1
      ],
      [
       3.0,
       49.7
      ],
      [
       3.5,
       51.6
      ],
      [
       4.0,
       50.8
      ],
      [
       4.5,
       48.8
      ],
      [
       5.0,
       48.1
      ],
      [
       5.5,
       48.4
      ],
      [
       6.0,
       51.7
      ],
      [
       6.5,
       52.7
      ],
      [
       7.0,
       54.2
      ],
      [
       7.5,
       53.7
      ],
      [
       8.0,
       49.3
      ],
      [
       8.5,
       46.5
      ],
      [
       9.0,
       39.0
      ],
      [
       9.5,
       39.9
      ],
      [
       10.0,
       39.9
      ],
      [
       10.5,
       39.4
      ],
      [
       11.0,
       40.3
      ]
     ]
    },
    {
     "id": 7,
     "cx": 0.1287,
     "cy": 0.506,
     "w": 0.0571,
     "h": 0.1557,
     "mean": 49.1,
     "low_pct": 53.8,
     "cue": "reading / writing",
     "n": 91,
     "series": [
      [
       2.0,
       40.5
      ],
      [
       2.5,
       39.9
      ],
      [
       3.0,
       40.9
      ],
      [
       3.5,
       42.9
      ],
      [
       4.0,
       46.2
      ],
      [
       4.5,
       48.8
      ],
      [
       5.0,
       49.2
      ],
      [
       5.5,
       50.6
      ],
      [
       6.0,
       52.9
      ],
      [
       6.5,
       54.7
      ],
      [
       7.0,
       55.0
      ],
      [
       7.5,
       55.1
      ],
      [
       8.0,
       54.1
      ],
      [
       8.5,
       53.4
      ],
      [
       9.0,
       51.2
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       49.6
      ],
      [
       10.5,
       48.9
      ],
      [
       11.0,
       49.5
      ]
     ]
    },
    {
     "id": 11,
     "cx": 0.3065,
     "cy": 0.51,
     "w": 0.044,
     "h": 0.1144,
     "mean": 49.4,
     "low_pct": 23.3,
     "cue": "reading / taking notes",
     "n": 86,
     "series": [
      [
       2.0,
       50.8
      ],
      [
       2.5,
       51.9
      ],
      [
       3.0,
       51.6
      ],
      [
       3.5,
       52.0
      ],
      [
       4.0,
       51.2
      ],
      [
       4.5,
       50.8
      ],
      [
       5.0,
       50.8
      ],
      [
       5.5,
       51.4
      ],
      [
       6.0,
       53.0
      ],
      [
       6.5,
       53.3
      ],
      [
       7.0,
       53.4
      ],
      [
       7.5,
       53.2
      ],
      [
       8.0,
       51.4
      ],
      [
       8.5,
       49.4
      ],
      [
       9.0,
       44.6
      ],
      [
       9.5,
       42.3
      ],
      [
       10.0,
       32.5
      ],
      [
       10.5,
       41.1
      ],
      [
       11.0,
       40.7
      ]
     ]
    },
    {
     "id": 12,
     "cx": 0.2453,
     "cy": 0.5764,
     "w": 0.0855,
     "h": 0.2029,
     "mean": 52.5,
     "low_pct": 22.0,
     "cue": "reading / writing",
     "n": 91,
     "series": [
      [
       2.0,
       55.4
      ],
      [
       2.5,
       54.5
      ],
      [
       3.0,
       53.7
      ],
      [
       3.5,
       54.6
      ],
      [
       4.0,
       53.2
      ],
      [
       4.5,
       51.3
      ],
      [
       5.0,
       50.8
      ],
      [
       5.5,
       51.0
      ],
      [
       6.0,
       54.2
      ],
      [
       6.5,
       55.5
      ],
      [
       7.0,
       57.1
      ],
      [
       7.5,
       56.8
      ],
      [
       8.0,
       54.5
      ],
      [
       8.5,
       52.5
      ],
      [
       9.0,
       49.3
      ],
      [
       9.5,
       47.2
      ],
      [
       10.0,
       46.8
      ],
      [
       10.5,
       47.4
      ],
      [
       11.0,
       47.5
      ]
     ]
    },
    {
     "id": 13,
     "cx": 0.0414,
     "cy": 0.5537,
     "w": 0.0773,
     "h": 0.2043,
     "mean": 56.9,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "n": 91,
     "series": [
      [
       2.0,
       55.4
      ],
      [
       2.5,
       56.4
      ],
      [
       3.0,
       55.8
      ],
      [
       3.5,
       56.4
      ],
      [
       4.0,
       56.1
      ],
      [
       4.5,
       55.3
      ],
      [
       5.0,
       55.3
      ],
      [
       5.5,
       55.5
      ],
      [
       6.0,
       58.1
      ],
      [
       6.5,
       59.4
      ],
      [
       7.0,
       60.5
      ],
      [
       7.5,
       60.5
      ],
      [
       8.0,
       59.9
      ],
      [
       8.5,
       58.7
      ],
      [
       9.0,
       57.9
      ],
      [
       9.5,
       57.1
      ],
      [
       10.0,
       55.7
      ],
      [
       10.5,
       52.2
      ],
      [
       11.0,
       50.7
      ]
     ]
    },
    {
     "id": 1,
     "cx": 0.1595,
     "cy": 0.4854,
     "w": 0.0343,
     "h": 0.1391,
     "mean": 43.5,
     "low_pct": 73.3,
     "cue": "reading / writing",
     "n": 90,
     "series": [
      [
       2.0,
       41.7
      ],
      [
       2.5,
       37.8
      ],
      [
       3.0,
       36.6
      ],
      [
       3.5,
       37.1
      ],
      [
       4.0,
       39.0
      ],
      [
       4.5,
       38.5
      ],
      [
       5.0,
       38.0
      ],
      [
       5.5,
       39.8
      ],
      [
       6.0,
       45.5
      ],
      [
       6.5,
       50.8
      ],
      [
       7.0,
       51.5
      ],
      [
       7.5,
       52.4
      ],
      [
       8.0,
       51.9
      ],
      [
       8.5,
       51.3
      ],
      [
       9.0,
       46.4
      ],
      [
       9.5,
       43.0
      ],
      [
       10.0,
       40.7
      ],
      [
       10.5,
       41.1
      ],
      [
       11.0,
       42.3
      ]
     ]
    }
   ],
   "class_mean": 48.8,
   "low_share": 44.86
  },
  {
   "key": "room_B_non_attentive_1",
   "title": "Seminar room B (non attentive 1)",
   "room": "room_B",
   "labelled": true,
   "video": "media/room_B_non_attentive_1.mp4",
   "poster": "media/room_B_non_attentive_1.jpg",
   "frame": [
    1280,
    720
   ],
   "duration": 10.0,
   "students_tracked": 19,
   "timeline": [
    [
     1.7,
     43.6,
     100.0,
     3
    ],
    [
     1.8,
     45.0,
     62.5,
     16
    ],
    [
     1.9,
     44.5,
     68.8,
     16
    ],
    [
     2.0,
     45.4,
     62.5,
     16
    ],
    [
     2.1,
     47.8,
     43.8,
     16
    ],
    [
     2.2,
     47.9,
     43.8,
     16
    ],
    [
     2.3,
     46.1,
     52.9,
     17
    ],
    [
     2.4,
     48.0,
     35.3,
     17
    ],
    [
     2.5,
     48.0,
     35.3,
     17
    ],
    [
     2.6,
     47.8,
     41.2,
     17
    ],
    [
     2.7,
     45.8,
     43.8,
     16
    ],
    [
     2.8,
     47.5,
     37.5,
     16
    ],
    [
     2.9,
     47.0,
     41.2,
     17
    ],
    [
     3.0,
     46.7,
     44.4,
     18
    ],
    [
     3.1,
     46.3,
     55.6,
     18
    ],
    [
     3.2,
     46.0,
     55.6,
     18
    ],
    [
     3.3,
     45.3,
     58.8,
     17
    ],
    [
     3.4,
     45.2,
     56.2,
     16
    ],
    [
     3.5,
     46.5,
     53.3,
     15
    ],
    [
     3.6,
     44.8,
     60.0,
     15
    ],
    [
     3.7,
     46.2,
     53.3,
     15
    ],
    [
     3.8,
     46.0,
     53.3,
     15
    ],
    [
     3.9,
     44.9,
     56.2,
     16
    ],
    [
     4.0,
     45.5,
     62.5,
     16
    ],
    [
     4.1,
     44.0,
     62.5,
     16
    ],
    [
     4.2,
     42.5,
     75.0,
     16
    ],
    [
     4.3,
     45.3,
     68.8,
     16
    ],
    [
     4.4,
     46.0,
     60.0,
     15
    ],
    [
     4.5,
     44.1,
     73.3,
     15
    ],
    [
     4.6,
     43.6,
     80.0,
     15
    ],
    [
     4.7,
     43.1,
     80.0,
     15
    ],
    [
     4.8,
     42.2,
     80.0,
     15
    ],
    [
     4.9,
     42.4,
     80.0,
     15
    ],
    [
     5.0,
     42.8,
     80.0,
     15
    ],
    [
     5.1,
     42.8,
     73.3,
     15
    ],
    [
     5.2,
     44.0,
     66.7,
     15
    ],
    [
     5.3,
     41.0,
     80.0,
     15
    ],
    [
     5.4,
     40.8,
     80.0,
     15
    ],
    [
     5.5,
     41.9,
     73.3,
     15
    ],
    [
     5.6,
     42.6,
     60.0,
     15
    ],
    [
     5.7,
     42.6,
     66.7,
     15
    ],
    [
     5.8,
     42.4,
     66.7,
     15
    ],
    [
     5.9,
     42.8,
     73.3,
     15
    ],
    [
     6.0,
     42.5,
     73.3,
     15
    ],
    [
     6.1,
     42.8,
     68.8,
     16
    ],
    [
     6.2,
     43.2,
     68.8,
     16
    ],
    [
     6.3,
     44.1,
     68.8,
     16
    ],
    [
     6.4,
     43.6,
     68.8,
     16
    ],
    [
     6.5,
     44.9,
     62.5,
     16
    ],
    [
     6.6,
     43.7,
     68.8,
     16
    ],
    [
     6.7,
     45.8,
     62.5,
     16
    ],
    [
     6.8,
     45.9,
     62.5,
     16
    ],
    [
     6.9,
     48.0,
     56.2,
     16
    ],
    [
     7.0,
     48.1,
     56.2,
     16
    ],
    [
     7.1,
     48.4,
     56.2,
     16
    ],
    [
     7.2,
     48.7,
     56.2,
     16
    ],
    [
     7.3,
     48.8,
     56.2,
     16
    ],
    [
     7.4,
     48.8,
     62.5,
     16
    ],
    [
     7.5,
     48.9,
     56.2,
     16
    ],
    [
     7.6,
     46.4,
     56.2,
     16
    ],
    [
     7.7,
     48.3,
     50.0,
     16
    ],
    [
     7.8,
     49.3,
     43.8,
     16
    ],
    [
     7.9,
     51.6,
     43.8,
     16
    ],
    [
     8.0,
     50.3,
     43.8,
     16
    ],
    [
     8.1,
     47.4,
     56.2,
     16
    ],
    [
     8.2,
     50.5,
     43.8,
     16
    ],
    [
     8.3,
     50.5,
     50.0,
     16
    ],
    [
     8.4,
     49.3,
     56.2,
     16
    ],
    [
     8.5,
     47.6,
     62.5,
     16
    ],
    [
     8.6,
     50.6,
     56.2,
     16
    ],
    [
     8.7,
     50.6,
     50.0,
     16
    ],
    [
     8.8,
     49.7,
     43.8,
     16
    ],
    [
     8.9,
     50.5,
     50.0,
     16
    ],
    [
     9.0,
     49.2,
     50.0,
     16
    ],
    [
     9.1,
     50.4,
     43.8,
     16
    ],
    [
     9.2,
     50.7,
     43.8,
     16
    ],
    [
     9.3,
     50.9,
     43.8,
     16
    ],
    [
     9.4,
     51.7,
     43.8,
     16
    ],
    [
     9.5,
     49.3,
     43.8,
     16
    ],
    [
     9.6,
     50.4,
     43.8,
     16
    ],
    [
     9.7,
     51.6,
     43.8,
     16
    ],
    [
     9.8,
     51.2,
     37.5,
     16
    ],
    [
     9.9,
     51.7,
     43.8,
     16
    ],
    [
     10.0,
     50.0,
     43.8,
     16
    ]
   ],
   "students": [
    {
     "id": 8,
     "n": 83,
     "mean": 30.4,
     "low_pct": 94.0,
     "cue": "using phone",
     "alerts": 1,
     "daisee": 71.1,
     "daisee_aux": [
      2.42,
      0.62,
      0.17,
      0.34
     ],
     "scb": "leaning on table",
     "scb_p": 0.25
    },
    {
     "id": 18,
     "n": 72,
     "mean": 33.8,
     "low_pct": 100.0,
     "cue": "glancing right",
     "alerts": 1,
     "daisee": 85.1,
     "daisee_aux": [
      2.5,
      0.39,
      0.11,
      0.33
     ],
     "scb": "leaning on table",
     "scb_p": 0.21
    },
    {
     "id": 5,
     "n": 22,
     "mean": 38.7,
     "low_pct": 100.0,
     "cue": "reading / taking notes",
     "alerts": 1,
     "daisee": 66.9,
     "daisee_aux": [
      2.43,
      0.84,
      0.09,
      0.3
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 12,
     "n": 83,
     "mean": 40.7,
     "low_pct": 43.4,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 67.9,
     "daisee_aux": [
      2.33,
      0.7,
      0.13,
      0.27
     ],
     "scb": "leaning on table",
     "scb_p": 0.44
    },
    {
     "id": 19,
     "n": 40,
     "mean": 42.8,
     "low_pct": 100.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 68.4,
     "daisee_aux": [
      2.39,
      0.51,
      0.16,
      0.31
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 15,
     "n": 83,
     "mean": 43.6,
     "low_pct": 84.3,
     "cue": "reading / taking notes",
     "alerts": 1,
     "daisee": 68.0,
     "daisee_aux": [
      2.37,
      0.62,
      0.18,
      0.35
     ],
     "scb": "hand-raising",
     "scb_p": 0.18
    },
    {
     "id": 7,
     "n": 83,
     "mean": 43.8,
     "low_pct": 66.3,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 70.0,
     "daisee_aux": [
      2.37,
      0.63,
      0.2,
      0.32
     ],
     "scb": "hand-raising",
     "scb_p": 0.18
    },
    {
     "id": 14,
     "n": 83,
     "mean": 45.6,
     "low_pct": 88.0,
     "cue": "thinking, chin on hand",
     "alerts": 0,
     "daisee": 99.5,
     "daisee_aux": [
      2.88,
      1.05,
      0.51,
      -0.03
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 10,
     "n": 83,
     "mean": 47.1,
     "low_pct": 56.6,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 67.2,
     "daisee_aux": [
      2.48,
      0.73,
      0.1,
      0.2
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 17,
     "n": 8,
     "mean": 47.5,
     "low_pct": 75.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 66.3,
     "daisee_aux": [
      2.48,
      0.89,
      0.07,
      0.33
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 16,
     "n": 84,
     "mean": 47.8,
     "low_pct": 73.8,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 67.6,
     "daisee_aux": [
      2.2,
      0.74,
      0.23,
      0.25
     ],
     "scb": "reading",
     "scb_p": 0.24
    },
    {
     "id": 2,
     "n": 16,
     "mean": 48.0,
     "low_pct": 56.2,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 67.8,
     "daisee_aux": [
      2.27,
      0.68,
      0.27,
      0.28
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 4,
     "n": 83,
     "mean": 49.2,
     "low_pct": 47.0,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 68.2,
     "daisee_aux": [
      2.42,
      0.56,
      0.12,
      0.26
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 9,
     "n": 83,
     "mean": 49.4,
     "low_pct": 67.5,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.1,
     "daisee_aux": [
      2.38,
      0.7,
      0.15,
      0.31
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 3,
     "n": 84,
     "mean": 51.6,
     "low_pct": 10.7,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 76.6,
     "daisee_aux": [
      2.39,
      0.43,
      -0.03,
      0.17
     ],
     "scb": "reading",
     "scb_p": 0.67
    },
    {
     "id": 1,
     "n": 83,
     "mean": 52.2,
     "low_pct": 20.5,
     "cue": "reading",
     "alerts": 1,
     "daisee": 95.8,
     "daisee_aux": [
      2.71,
      0.94,
      0.31,
      0.05
     ],
     "scb": "reading",
     "scb_p": 0.54
    },
    {
     "id": 13,
     "n": 83,
     "mean": 53.6,
     "low_pct": 31.3,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 68.1,
     "daisee_aux": [
      2.39,
      0.68,
      0.13,
      0.36
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 6,
     "n": 83,
     "mean": 55.7,
     "low_pct": 30.1,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 65.5,
     "daisee_aux": [
      2.23,
      0.95,
      0.01,
      0.11
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 11,
     "n": 83,
     "mean": 56.9,
     "low_pct": 14.5,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 68.8,
     "daisee_aux": [
      2.3,
      0.46,
      0.18,
      0.32
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    }
   ],
   "alerts": [
    {
     "t": 4.2,
     "student": 8,
     "kind": "sustained",
     "score": 22.0,
     "cue": "using phone",
     "recommendation": "Apply non-verbal proximity control: circulate calmly down the aisle near the flagged desk without halting instructional delivery.",
     "suggestion": "Avoid public confrontation; private physical presence typically causes the student to stow the device naturally."
    },
    {
     "t": 4.2,
     "student": 7,
     "kind": "sustained",
     "score": 32.9,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 4.2,
     "student": 15,
     "kind": "sustained",
     "score": 38.0,
     "cue": "reading / taking notes",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 4.2,
     "student": 5,
     "kind": "sustained",
     "score": 39.0,
     "cue": "reading / taking notes",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 4.2,
     "student": 10,
     "kind": "sustained",
     "score": 34.0,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 4.8,
     "student": 1,
     "kind": "sudden",
     "score": 42.1,
     "cue": "reading",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Wait one more cycle; brief dips are common."
    },
    {
     "t": 5.3,
     "student": 12,
     "kind": "sudden",
     "score": 39.9,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Wait one more cycle; brief dips are common."
    },
    {
     "t": 5.4,
     "student": 18,
     "kind": "sustained",
     "score": 33.1,
     "cue": "glancing right",
     "recommendation": "Direct an inclusive, low-stakes retrieval question toward that sector of the room, or use a vocal inflection change to refocus eyes on the visual aid.",
     "suggestion": "Circulate toward that side of the room and re-anchor visual engagement."
    },
    {
     "t": 5.8,
     "student": 6,
     "kind": "sudden",
     "score": 44.2,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Wait one more cycle; brief dips are common."
    },
    {
     "t": 8.8,
     "student": 4,
     "kind": "sudden",
     "score": 40.9,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Wait one more cycle; brief dips are common."
    }
   ],
   "seats": [
    {
     "id": 16,
     "cx": 0.289,
     "cy": 0.482,
     "w": 0.0297,
     "h": 0.0446,
     "mean": 47.8,
     "low_pct": 73.8,
     "cue": "reading / taking notes",
     "n": 84,
     "series": [
      [
       1.5,
       46.1
      ],
      [
       2.0,
       49.2
      ],
      [
       2.5,
       50.6
      ],
      [
       3.0,
       50.3
      ],
      [
       3.5,
       48.9
      ],
      [
       4.0,
       47.3
      ],
      [
       4.5,
       45.6
      ],
      [
       5.0,
       45.9
      ],
      [
       5.5,
       46.9
      ],
      [
       6.0,
       46.9
      ],
      [
       6.5,
       47.1
      ],
      [
       7.0,
       48.5
      ],
      [
       7.5,
       48.7
      ],
      [
       8.0,
       50.8
      ],
      [
       8.5,
       50.3
      ],
      [
       9.0,
       49.7
      ],
      [
       9.5,
       43.5
      ],
      [
       10.0,
       39.8
      ]
     ]
    },
    {
     "id": 2,
     "cx": 0.5233,
     "cy": 0.6628,
     "w": 0.0358,
     "h": 0.0807,
     "mean": 48.0,
     "low_pct": 56.2,
     "cue": "reading / taking notes",
     "n": 16,
     "series": [
      [
       1.5,
       36.8
      ],
      [
       2.0,
       48.9
      ],
      [
       2.5,
       51.5
      ],
      [
       3.0,
       45.8
      ]
     ]
    },
    {
     "id": 3,
     "cx": 0.6351,
     "cy": 0.4888,
     "w": 0.0423,
     "h": 0.0859,
     "mean": 51.6,
     "low_pct": 10.7,
     "cue": "reading / taking notes",
     "n": 84,
     "series": [
      [
       1.5,
       47.9
      ],
      [
       2.0,
       48.2
      ],
      [
       2.5,
       48.9
      ],
      [
       3.0,
       50.1
      ],
      [
       3.5,
       52.9
      ],
      [
       4.0,
       53.9
      ],
      [
       4.5,
       53.1
      ],
      [
       5.0,
       50.6
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       51.5
      ],
      [
       6.5,
       55.1
      ],
      [
       7.0,
       56.0
      ],
      [
       7.5,
       54.3
      ],
      [
       8.0,
       53.6
      ],
      [
       8.5,
       52.0
      ],
      [
       9.0,
       50.7
      ],
      [
       9.5,
       46.1
      ],
      [
       10.0,
       51.4
      ]
     ]
    },
    {
     "id": 1,
     "cx": 0.8557,
     "cy": 0.6165,
     "w": 0.0556,
     "h": 0.1667,
     "mean": 52.2,
     "low_pct": 20.5,
     "cue": "reading",
     "n": 83,
     "series": [
      [
       2.0,
       61.2
      ],
      [
       2.5,
       61.6
      ],
      [
       3.0,
       62.1
      ],
      [
       3.5,
       63.2
      ],
      [
       4.0,
       55.5
      ],
      [
       4.5,
       42.5
      ],
      [
       5.0,
       39.4
      ],
      [
       5.5,
       43.5
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.0
      ],
      [
       7.0,
       50.0
      ],
      [
       7.5,
       50.0
      ],
      [
       8.0,
       50.0
      ],
      [
       8.5,
       47.6
      ],
      [
       9.0,
       50.8
      ],
      [
       9.5,
       53.2
      ],
      [
       10.0,
       58.2
      ]
     ]
    },
    {
     "id": 4,
     "cx": 0.128,
     "cy": 0.5332,
     "w": 0.0374,
     "h": 0.0961,
     "mean": 49.2,
     "low_pct": 47.0,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       2.0,
       54.4
      ],
      [
       2.5,
       47.5
      ],
      [
       3.0,
       52.3
      ],
      [
       3.5,
       41.5
      ],
      [
       4.0,
       41.2
      ],
      [
       4.5,
       47.2
      ],
      [
       5.0,
       55.3
      ],
      [
       5.5,
       58.3
      ],
      [
       6.0,
       60.7
      ],
      [
       6.5,
       56.3
      ],
      [
       7.0,
       62.6
      ],
      [
       7.5,
       53.1
      ],
      [
       8.0,
       48.2
      ],
      [
       8.5,
       41.3
      ],
      [
       9.0,
       37.6
      ],
      [
       9.5,
       36.3
      ],
      [
       10.0,
       37.7
      ]
     ]
    },
    {
     "id": 8,
     "cx": 0.0976,
     "cy": 0.679,
     "w": 0.0929,
     "h": 0.2643,
     "mean": 30.4,
     "low_pct": 94.0,
     "cue": "using phone",
     "n": 83,
     "series": [
      [
       2.0,
       22.0
      ],
      [
       2.5,
       22.0
      ],
      [
       3.0,
       22.0
      ],
      [
       3.5,
       22.0
      ],
      [
       4.0,
       22.0
      ],
      [
       4.5,
       22.0
      ],
      [
       5.0,
       22.0
      ],
      [
       5.5,
       22.0
      ],
      [
       6.0,
       22.0
      ],
      [
       6.5,
       22.0
      ],
      [
       7.0,
       47.3
      ],
      [
       7.5,
       48.6
      ],
      [
       8.0,
       43.6
      ],
      [
       8.5,
       42.6
      ],
      [
       9.0,
       39.1
      ],
      [
       9.5,
       39.4
      ],
      [
       10.0,
       38.8
      ]
     ]
    },
    {
     "id": 9,
     "cx": 0.0754,
     "cy": 0.5295,
     "w": 0.049,
     "h": 0.1514,
     "mean": 49.4,
     "low_pct": 67.5,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       2.0,
       55.3
      ],
      [
       2.5,
       56.4
      ],
      [
       3.0,
       56.1
      ],
      [
       3.5,
       57.1
      ],
      [
       4.0,
       53.5
      ],
      [
       4.5,
       49.6
      ],
      [
       5.0,
       45.7
      ],
      [
       5.5,
       45.1
      ],
      [
       6.0,
       45.6
      ],
      [
       6.5,
       45.7
      ],
      [
       7.0,
       46.5
      ],
      [
       7.5,
       45.6
      ],
      [
       8.0,
       46.4
      ],
      [
       8.5,
       46.4
      ],
      [
       9.0,
       47.2
      ],
      [
       9.5,
       49.0
      ],
      [
       10.0,
       49.2
      ]
     ]
    },
    {
     "id": 7,
     "cx": 0.5563,
     "cy": 0.5219,
     "w": 0.0596,
     "h": 0.1653,
     "mean": 43.8,
     "low_pct": 66.3,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       2.0,
       31.0
      ],
      [
       2.5,
       29.8
      ],
      [
       3.0,
       29.3
      ],
      [
       3.5,
       30.7
      ],
      [
       4.0,
       32.7
      ],
      [
       4.5,
       44.8
      ],
      [
       5.0,
       38.0
      ],
      [
       5.5,
       42.5
      ],
      [
       6.0,
       44.2
      ],
      [
       6.5,
       44.0
      ],
      [
       7.0,
       47.7
      ],
      [
       7.5,
       49.8
      ],
      [
       8.0,
       53.0
      ],
      [
       8.5,
       57.8
      ],
      [
       9.0,
       57.8
      ],
      [
       9.5,
       58.4
      ],
      [
       10.0,
       58.6
      ]
     ]
    },
    {
     "id": 15,
     "cx": 0.478,
     "cy": 0.5111,
     "w": 0.0418,
     "h": 0.1105,
     "mean": 43.6,
     "low_pct": 84.3,
     "cue": "reading / taking notes",
     "n": 83,
     "series": [
      [
       2.0,
       34.1
      ],
      [
       2.5,
       36.4
      ],
      [
       3.0,
       38.9
      ],
      [
       3.5,
       39.0
      ],
      [
       4.0,
       37.8
      ],
      [
       4.5,
       43.8
      ],
      [
       5.0,
       42.4
      ],
      [
       5.5,
       45.2
      ],
      [
       6.0,
       44.3
      ],
      [
       6.5,
       45.8
      ],
      [
       7.0,
       44.7
      ],
      [
       7.5,
       45.7
      ],
      [
       8.0,
       44.3
      ],
      [
       8.5,
       48.0
      ],
      [
       9.0,
       50.2
      ],
      [
       9.5,
       52.4
      ],
      [
       10.0,
       50.9
      ]
     ]
    },
    {
     "id": 11,
     "cx": 0.3663,
     "cy": 0.6282,
     "w": 0.0505,
     "h": 0.1701,
     "mean": 56.9,
     "low_pct": 14.5,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       2.0,
       54.7
      ],
      [
       2.5,
       60.0
      ],
      [
       3.0,
       57.6
      ],
      [
       3.5,
       50.7
      ],
      [
       4.0,
       51.7
      ],
      [
       4.5,
       49.3
      ],
      [
       5.0,
       50.3
      ],
      [
       5.5,
       52.6
      ],
      [
       6.0,
       56.4
      ],
      [
       6.5,
       57.4
      ],
      [
       7.0,
       59.0
      ],
      [
       7.5,
       61.6
      ],
      [
       8.0,
       56.4
      ],
      [
       8.5,
       63.2
      ],
      [
       9.0,
       64.2
      ],
      [
       9.5,
       62.8
      ],
      [
       10.0,
       59.7
      ]
     ]
    },
    {
     "id": 6,
     "cx": 0.7858,
     "cy": 0.5488,
     "w": 0.0631,
     "h": 0.1624,
     "mean": 55.7,
     "low_pct": 30.1,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       2.0,
       59.5
      ],
      [
       2.5,
       59.3
      ],
      [
       3.0,
       58.9
      ],
      [
       3.5,
       63.1
      ],
      [
       4.0,
       65.3
      ],
      [
       4.5,
       60.7
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       44.5
      ],
      [
       6.0,
       40.7
      ],
      [
       6.5,
       41.6
      ],
      [
       7.0,
       44.9
      ],
      [
       7.5,
       51.1
      ],
      [
       8.0,
       58.9
      ],
      [
       8.5,
       61.1
      ],
      [
       9.0,
       62.9
      ],
      [
       9.5,
       64.3
      ],
      [
       10.0,
       63.6
      ]
     ]
    },
    {
     "id": 5,
     "cx": 0.4454,
     "cy": 0.4919,
     "w": 0.0405,
     "h": 0.1275,
     "mean": 38.7,
     "low_pct": 100.0,
     "cue": "reading / taking notes",
     "n": 22,
     "series": [
      [
       2.0,
       38.4
      ],
      [
       2.5,
       38.3
      ],
      [
       3.0,
       39.0
      ],
      [
       3.5,
       39.1
      ],
      [
       4.0,
       39.1
      ],
      [
       4.5,
       38.9
      ]
     ]
    },
    {
     "id": 13,
     "cx": 0.6817,
     "cy": 0.4921,
     "w": 0.0491,
     "h": 0.1357,
     "mean": 53.6,
     "low_pct": 31.3,
     "cue": "reading / taking notes",
     "n": 83,
     "series": [
      [
       2.0,
       46.8
      ],
      [
       2.5,
       50.9
      ],
      [
       3.0,
       54.8
      ],
      [
       3.5,
       54.9
      ],
      [
       4.0,
       50.6
      ],
      [
       4.5,
       46.3
      ],
      [
       5.0,
       44.3
      ],
      [
       5.5,
       46.3
      ],
      [
       6.0,
       49.2
      ],
      [
       6.5,
       56.2
      ],
      [
       7.0,
       58.5
      ],
      [
       7.5,
       61.3
      ],
      [
       8.0,
       60.5
      ],
      [
       8.5,
       58.9
      ],
      [
       9.0,
       57.7
      ],
      [
       9.5,
       57.4
      ],
      [
       10.0,
       58.1
      ]
     ]
    },
    {
     "id": 10,
     "cx": 0.3199,
     "cy": 0.5761,
     "w": 0.0405,
     "h": 0.1374,
     "mean": 47.1,
     "low_pct": 56.6,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       2.0,
       30.0
      ],
      [
       2.5,
       30.0
      ],
      [
       3.0,
       30.0
      ],
      [
       3.5,
       30.0
      ],
      [
       4.0,
       36.4
      ],
      [
       4.5,
       44.0
      ],
      [
       5.0,
       35.8
      ],
      [
       5.5,
       37.8
      ],
      [
       6.0,
       40.1
      ],
      [
       6.5,
       52.5
      ],
      [
       7.0,
       59.1
      ],
      [
       7.5,
       61.6
      ],
      [
       8.0,
       63.3
      ],
      [
       8.5,
       64.0
      ],
      [
       9.0,
       64.3
      ],
      [
       9.5,
       64.4
      ],
      [
       10.0,
       65.6
      ]
     ]
    },
    {
     "id": 12,
     "cx": 0.7082,
     "cy": 0.5958,
     "w": 0.0517,
     "h": 0.1101,
     "mean": 40.7,
     "low_pct": 43.4,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       2.0,
       54.8
      ],
      [
       2.5,
       56.1
      ],
      [
       3.0,
       59.8
      ],
      [
       3.5,
       57.5
      ],
      [
       4.0,
       54.3
      ],
      [
       4.5,
       48.1
      ],
      [
       5.0,
       47.0
      ],
      [
       5.5,
       16.8
      ],
      [
       6.0,
       16.0
      ],
      [
       6.5,
       16.0
      ],
      [
       7.0,
       16.0
      ],
      [
       7.5,
       16.0
      ],
      [
       8.0,
       44.1
      ],
      [
       8.5,
       43.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       51.0
      ]
     ]
    },
    {
     "id": 14,
     "cx": 0.2672,
     "cy": 0.5475,
     "w": 0.047,
     "h": 0.1329,
     "mean": 45.6,
     "low_pct": 88.0,
     "cue": "thinking, chin on hand",
     "n": 83,
     "series": [
      [
       2.0,
       49.3
      ],
      [
       2.5,
       53.2
      ],
      [
       3.0,
       50.2
      ],
      [
       3.5,
       43.8
      ],
      [
       4.0,
       42.1
      ],
      [
       4.5,
       41.3
      ],
      [
       5.0,
       43.1
      ],
      [
       5.5,
       41.7
      ],
      [
       6.0,
       42.1
      ],
      [
       6.5,
       42.2
      ],
      [
       7.0,
       42.3
      ],
      [
       7.5,
       43.0
      ],
      [
       8.0,
       46.3
      ],
      [
       8.5,
       49.3
      ],
      [
       9.0,
       49.6
      ],
      [
       9.5,
       48.7
      ],
      [
       10.0,
       48.6
      ]
     ]
    },
    {
     "id": 17,
     "cx": 0.4233,
     "cy": 0.6883,
     "w": 0.0493,
     "h": 0.1146,
     "mean": 47.5,
     "low_pct": 75.0,
     "cue": "reading / writing",
     "n": 8,
     "series": [
      [
       2.5,
       49.8
      ],
      [
       3.0,
       45.7
      ],
      [
       3.5,
       43.7
      ]
     ]
    },
    {
     "id": 18,
     "cx": 0.4518,
     "cy": 0.6406,
     "w": 0.0413,
     "h": 0.0878,
     "mean": 33.8,
     "low_pct": 100.0,
     "cue": "glancing right",
     "n": 72,
     "series": [
      [
       3.0,
       34.3
      ],
      [
       3.5,
       31.5
      ],
      [
       4.0,
       28.7
      ],
      [
       4.5,
       29.1
      ],
      [
       5.0,
       32.6
      ],
      [
       5.5,
       33.6
      ],
      [
       6.0,
       31.4
      ],
      [
       6.5,
       34.6
      ],
      [
       7.0,
       36.9
      ],
      [
       7.5,
       41.9
      ],
      [
       8.0,
       36.8
      ],
      [
       8.5,
       30.0
      ],
      [
       9.0,
       30.0
      ],
      [
       9.5,
       39.2
      ],
      [
       10.0,
       39.6
      ]
     ]
    },
    {
     "id": 19,
     "cx": 0.4577,
     "cy": 0.4754,
     "w": 0.0256,
     "h": 0.091,
     "mean": 42.8,
     "low_pct": 100.0,
     "cue": "reading / writing",
     "n": 40,
     "series": [
      [
       6.0,
       43.8
      ],
      [
       6.5,
       44.6
      ],
      [
       7.0,
       45.6
      ],
      [
       7.5,
       39.6
      ],
      [
       8.0,
       40.9
      ],
      [
       8.5,
       39.5
      ],
      [
       9.0,
       40.3
      ],
      [
       9.5,
       47.4
      ],
      [
       10.0,
       44.9
      ]
     ]
    }
   ],
   "class_mean": 46.55,
   "low_share": 57.74
  },
  {
   "key": "room_C_test_8",
   "title": "Classroom C (Test 8)",
   "room": "room_C",
   "labelled": true,
   "video": "media/room_C_test_8.mp4",
   "poster": "media/room_C_test_8.jpg",
   "frame": [
    1280,
    720
   ],
   "duration": 9.917,
   "students_tracked": 21,
   "timeline": [
    [
     1.71,
     43.5,
     38.5,
     13
    ],
    [
     1.83,
     45.4,
     38.1,
     21
    ],
    [
     1.92,
     45.9,
     38.1,
     21
    ],
    [
     2.0,
     45.4,
     38.1,
     21
    ],
    [
     2.12,
     45.7,
     38.1,
     21
    ],
    [
     2.21,
     45.7,
     38.1,
     21
    ],
    [
     2.33,
     46.8,
     28.6,
     21
    ],
    [
     2.42,
     47.6,
     23.8,
     21
    ],
    [
     2.5,
     46.0,
     28.6,
     21
    ],
    [
     2.62,
     46.1,
     28.6,
     21
    ],
    [
     2.71,
     46.7,
     33.3,
     21
    ],
    [
     2.83,
     46.4,
     33.3,
     21
    ],
    [
     2.92,
     47.3,
     33.3,
     21
    ],
    [
     3.0,
     46.2,
     33.3,
     21
    ],
    [
     3.12,
     46.4,
     33.3,
     21
    ],
    [
     3.21,
     46.1,
     33.3,
     21
    ],
    [
     3.33,
     44.5,
     38.1,
     21
    ],
    [
     3.42,
     45.8,
     33.3,
     21
    ],
    [
     3.5,
     46.2,
     28.6,
     21
    ],
    [
     3.62,
     46.1,
     28.6,
     21
    ],
    [
     3.71,
     44.5,
     38.1,
     21
    ],
    [
     3.83,
     45.0,
     33.3,
     21
    ],
    [
     3.92,
     44.5,
     33.3,
     21
    ],
    [
     4.0,
     44.0,
     38.1,
     21
    ],
    [
     4.12,
     44.2,
     38.1,
     21
    ],
    [
     4.21,
     44.1,
     38.1,
     21
    ],
    [
     4.33,
     44.3,
     38.1,
     21
    ],
    [
     4.42,
     44.5,
     38.1,
     21
    ],
    [
     4.5,
     44.2,
     38.1,
     21
    ],
    [
     4.62,
     44.0,
     47.6,
     21
    ],
    [
     4.71,
     43.5,
     52.4,
     21
    ],
    [
     4.83,
     43.4,
     52.4,
     21
    ],
    [
     4.92,
     43.9,
     47.6,
     21
    ],
    [
     5.0,
     43.6,
     52.4,
     21
    ],
    [
     5.12,
     43.7,
     47.6,
     21
    ],
    [
     5.21,
     44.0,
     47.6,
     21
    ],
    [
     5.33,
     43.1,
     52.4,
     21
    ],
    [
     5.42,
     44.4,
     42.9,
     21
    ],
    [
     5.5,
     45.3,
     42.9,
     21
    ],
    [
     5.62,
     45.4,
     47.6,
     21
    ],
    [
     5.71,
     45.2,
     47.6,
     21
    ],
    [
     5.83,
     45.1,
     52.4,
     21
    ],
    [
     5.92,
     44.1,
     57.1,
     21
    ],
    [
     6.0,
     45.7,
     52.4,
     21
    ],
    [
     6.12,
     46.2,
     47.6,
     21
    ],
    [
     6.21,
     46.1,
     52.4,
     21
    ],
    [
     6.33,
     45.9,
     52.4,
     21
    ],
    [
     6.42,
     46.3,
     52.4,
     21
    ],
    [
     6.5,
     45.9,
     52.4,
     21
    ],
    [
     6.62,
     45.7,
     52.4,
     21
    ],
    [
     6.71,
     45.4,
     52.4,
     21
    ],
    [
     6.83,
     45.2,
     47.6,
     21
    ],
    [
     6.92,
     45.7,
     47.6,
     21
    ],
    [
     7.0,
     45.7,
     47.6,
     21
    ],
    [
     7.12,
     46.0,
     47.6,
     21
    ],
    [
     7.21,
     46.1,
     47.6,
     21
    ],
    [
     7.33,
     44.9,
     52.4,
     21
    ],
    [
     7.42,
     45.3,
     50.0,
     20
    ],
    [
     7.5,
     46.5,
     45.0,
     20
    ],
    [
     7.62,
     46.3,
     45.0,
     20
    ],
    [
     7.71,
     47.1,
     45.0,
     20
    ],
    [
     7.83,
     47.1,
     45.0,
     20
    ],
    [
     7.92,
     47.3,
     35.0,
     20
    ],
    [
     8.0,
     47.7,
     42.1,
     19
    ],
    [
     8.12,
     47.8,
     36.8,
     19
    ],
    [
     8.21,
     49.5,
     31.6,
     19
    ],
    [
     8.33,
     49.9,
     31.6,
     19
    ],
    [
     8.42,
     50.4,
     26.3,
     19
    ],
    [
     8.5,
     50.8,
     26.3,
     19
    ],
    [
     8.62,
     51.0,
     26.3,
     19
    ],
    [
     8.71,
     51.1,
     26.3,
     19
    ],
    [
     8.83,
     51.2,
     26.3,
     19
    ],
    [
     8.92,
     52.5,
     21.1,
     19
    ],
    [
     9.0,
     52.1,
     20.0,
     20
    ],
    [
     9.12,
     52.2,
     20.0,
     20
    ],
    [
     9.21,
     52.4,
     20.0,
     20
    ],
    [
     9.33,
     52.0,
     20.0,
     20
    ],
    [
     9.42,
     52.7,
     20.0,
     20
    ],
    [
     9.5,
     53.1,
     15.0,
     20
    ],
    [
     9.62,
     52.9,
     15.0,
     20
    ],
    [
     9.71,
     53.1,
     15.0,
     20
    ],
    [
     9.83,
     53.8,
     15.0,
     20
    ],
    [
     9.92,
     54.1,
     10.0,
     20
    ]
   ],
   "students": [
    {
     "id": 21,
     "n": 57,
     "mean": 27.4,
     "low_pct": 100.0,
     "cue": "glancing right",
     "alerts": 1,
     "daisee": 79.3,
     "daisee_aux": [
      2.4,
      0.29,
      0.15,
      0.29
     ],
     "scb": "reading",
     "scb_p": 0.34
    },
    {
     "id": 2,
     "n": 82,
     "mean": 29.4,
     "low_pct": 98.8,
     "cue": "using phone",
     "alerts": 1,
     "daisee": 67.8,
     "daisee_aux": [
      2.34,
      0.7,
      0.08,
      0.3
     ],
     "scb": "reading",
     "scb_p": 0.24
    },
    {
     "id": 5,
     "n": 82,
     "mean": 37.0,
     "low_pct": 81.7,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 75.4,
     "daisee_aux": [
      2.31,
      0.44,
      0.15,
      0.33
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 8,
     "n": 83,
     "mean": 37.5,
     "low_pct": 94.0,
     "cue": "watching instructor",
     "alerts": 1,
     "daisee": 77.3,
     "daisee_aux": [
      2.45,
      0.48,
      0.06,
      0.26
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 11,
     "n": 83,
     "mean": 40.2,
     "low_pct": 73.5,
     "cue": "watching instructor",
     "alerts": 1,
     "daisee": 89.4,
     "daisee_aux": [
      2.83,
      -0.07,
      0.26,
      0.02
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 7,
     "n": 83,
     "mean": 40.9,
     "low_pct": 74.7,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 78.8,
     "daisee_aux": [
      2.55,
      0.39,
      0.18,
      0.21
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 19,
     "n": 73,
     "mean": 42.8,
     "low_pct": 94.5,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.0,
     "daisee_aux": [
      2.34,
      0.86,
      0.09,
      0.24
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 3,
     "n": 82,
     "mean": 46.1,
     "low_pct": 51.2,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.6,
     "daisee_aux": [
      2.32,
      0.86,
      0.2,
      0.27
     ],
     "scb": "hand-raising",
     "scb_p": 0.4
    },
    {
     "id": 18,
     "n": 83,
     "mean": 46.7,
     "low_pct": 41.0,
     "cue": "watching instructor",
     "alerts": 0,
     "daisee": 78.9,
     "daisee_aux": [
      2.32,
      0.5,
      0.06,
      0.33
     ],
     "scb": "reading",
     "scb_p": 0.41
    },
    {
     "id": 6,
     "n": 82,
     "mean": 47.5,
     "low_pct": 35.4,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 76.7,
     "daisee_aux": [
      2.38,
      0.93,
      0.09,
      0.11
     ],
     "scb": "writing",
     "scb_p": 0.39
    },
    {
     "id": 10,
     "n": 82,
     "mean": 49.3,
     "low_pct": 28.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 81.1,
     "daisee_aux": [
      2.38,
      0.92,
      0.21,
      0.18
     ],
     "scb": "reading",
     "scb_p": 0.34
    },
    {
     "id": 15,
     "n": 83,
     "mean": 49.6,
     "low_pct": 25.3,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 90.8,
     "daisee_aux": [
      2.69,
      0.62,
      0.11,
      0.21
     ],
     "scb": "reading",
     "scb_p": 0.55
    },
    {
     "id": 1,
     "n": 82,
     "mean": 49.9,
     "low_pct": 9.8,
     "cue": "reading",
     "alerts": 0,
     "daisee": 96.6,
     "daisee_aux": [
      2.89,
      0.92,
      0.33,
      0.12
     ],
     "scb": "reading",
     "scb_p": 0.87
    },
    {
     "id": 12,
     "n": 83,
     "mean": 50.5,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 79.6,
     "daisee_aux": [
      2.33,
      0.52,
      -0.01,
      0.2
     ],
     "scb": "reading",
     "scb_p": 0.61
    },
    {
     "id": 14,
     "n": 83,
     "mean": 51.9,
     "low_pct": 6.0,
     "cue": "reading",
     "alerts": 0,
     "daisee": 80.8,
     "daisee_aux": [
      2.31,
      0.42,
      -0.06,
      0.11
     ],
     "scb": "reading",
     "scb_p": 0.85
    },
    {
     "id": 13,
     "n": 83,
     "mean": 52.9,
     "low_pct": 6.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 80.0,
     "daisee_aux": [
      2.5,
      0.76,
      0.13,
      0.16
     ],
     "scb": "reading",
     "scb_p": 0.44
    },
    {
     "id": 20,
     "n": 83,
     "mean": 53.7,
     "low_pct": 1.2,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 68.4,
     "daisee_aux": [
      2.34,
      0.8,
      0.06,
      0.14
     ],
     "scb": "reading",
     "scb_p": 0.26
    },
    {
     "id": 16,
     "n": 82,
     "mean": 54.1,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 90.8,
     "daisee_aux": [
      2.46,
      0.28,
      0.26,
      0.19
     ],
     "scb": "reading",
     "scb_p": 0.52
    },
    {
     "id": 9,
     "n": 83,
     "mean": 55.3,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.9,
     "daisee_aux": [
      2.48,
      0.68,
      0.09,
      0.34
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 4,
     "n": 82,
     "mean": 56.5,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 68.3,
     "daisee_aux": [
      2.38,
      0.67,
      0.14,
      0.32
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 17,
     "n": 83,
     "mean": 56.6,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.5,
     "daisee_aux": [
      2.31,
      0.62,
      0.27,
      0.33
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    }
   ],
   "alerts": [
    {
     "t": 3.9,
     "student": 11,
     "kind": "sudden",
     "score": 33.3,
     "cue": "watching instructor",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Wait one more cycle; brief dips are common."
    },
    {
     "t": 4.1,
     "student": 7,
     "kind": "sustained",
     "score": 27.0,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 4.1,
     "student": 21,
     "kind": "sustained",
     "score": 22.8,
     "cue": "glancing right",
     "recommendation": "Direct an inclusive, low-stakes retrieval question toward that sector of the room, or use a vocal inflection change to refocus eyes on the visual aid.",
     "suggestion": "Circulate toward that side of the room and re-anchor visual engagement."
    },
    {
     "t": 4.1,
     "student": 8,
     "kind": "sustained",
     "score": 30.2,
     "cue": "watching instructor",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 4.3,
     "student": 2,
     "kind": "sustained",
     "score": 22.0,
     "cue": "using phone",
     "recommendation": "Apply non-verbal proximity control: circulate calmly down the aisle near the flagged desk without halting instructional delivery.",
     "suggestion": "Avoid public confrontation; private physical presence typically causes the student to stow the device naturally."
    },
    {
     "t": 5.5,
     "student": 5,
     "kind": "sudden",
     "score": 24.8,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 6.3,
     "student": 6,
     "kind": "sudden",
     "score": 36.7,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Wait one more cycle; brief dips are common."
    }
   ],
   "seats": [
    {
     "id": 14,
     "cx": 0.8624,
     "cy": 0.6403,
     "w": 0.0648,
     "h": 0.1746,
     "mean": 51.9,
     "low_pct": 6.0,
     "cue": "reading",
     "n": 83,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       51.8
      ],
      [
       3.0,
       53.2
      ],
      [
       3.5,
       54.0
      ],
      [
       4.0,
       55.1
      ],
      [
       4.5,
       55.4
      ],
      [
       5.0,
       51.5
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       46.0
      ],
      [
       6.5,
       50.0
      ],
      [
       7.0,
       50.0
      ],
      [
       7.5,
       50.9
      ],
      [
       8.0,
       55.7
      ],
      [
       8.5,
       58.8
      ],
      [
       9.0,
       58.7
      ],
      [
       9.5,
       46.0
      ],
      [
       10.0,
       35.6
      ]
     ]
    },
    {
     "id": 11,
     "cx": 0.1797,
     "cy": 0.4427,
     "w": 0.0567,
     "h": 0.1419,
     "mean": 40.2,
     "low_pct": 73.5,
     "cue": "watching instructor",
     "n": 83,
     "series": [
      [
       1.5,
       53.9
      ],
      [
       2.0,
       54.1
      ],
      [
       2.5,
       54.1
      ],
      [
       3.0,
       53.6
      ],
      [
       3.5,
       35.3
      ],
      [
       4.0,
       30.0
      ],
      [
       4.5,
       30.0
      ],
      [
       5.0,
       30.0
      ],
      [
       5.5,
       41.5
      ],
      [
       6.0,
       42.8
      ],
      [
       6.5,
       40.0
      ],
      [
       7.0,
       34.8
      ],
      [
       7.5,
       33.8
      ],
      [
       8.0,
       31.6
      ],
      [
       8.5,
       34.3
      ],
      [
       9.0,
       39.5
      ],
      [
       9.5,
       49.1
      ],
      [
       10.0,
       54.4
      ]
     ]
    },
    {
     "id": 9,
     "cx": 0.5989,
     "cy": 0.4213,
     "w": 0.0568,
     "h": 0.1516,
     "mean": 55.3,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       1.5,
       56.2
      ],
      [
       2.0,
       56.3
      ],
      [
       2.5,
       55.8
      ],
      [
       3.0,
       55.5
      ],
      [
       3.5,
       55.6
      ],
      [
       4.0,
       55.2
      ],
      [
       4.5,
       55.5
      ],
      [
       5.0,
       54.4
      ],
      [
       5.5,
       54.1
      ],
      [
       6.0,
       54.9
      ],
      [
       6.5,
       54.8
      ],
      [
       7.0,
       54.7
      ],
      [
       7.5,
       54.1
      ],
      [
       8.0,
       54.1
      ],
      [
       8.5,
       55.4
      ],
      [
       9.0,
       56.5
      ],
      [
       9.5,
       57.3
      ],
      [
       10.0,
       58.5
      ]
     ]
    },
    {
     "id": 7,
     "cx": 0.5409,
     "cy": 0.3392,
     "w": 0.0398,
     "h": 0.1065,
     "mean": 40.9,
     "low_pct": 74.7,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       1.5,
       29.7
      ],
      [
       2.0,
       31.7
      ],
      [
       2.5,
       33.6
      ],
      [
       3.0,
       36.2
      ],
      [
       3.5,
       28.6
      ],
      [
       4.0,
       26.4
      ],
      [
       4.5,
       26.8
      ],
      [
       5.0,
       31.6
      ],
      [
       5.5,
       36.3
      ],
      [
       6.0,
       41.5
      ],
      [
       6.5,
       42.4
      ],
      [
       7.0,
       43.1
      ],
      [
       7.5,
       45.4
      ],
      [
       8.0,
       52.4
      ],
      [
       8.5,
       58.0
      ],
      [
       9.0,
       58.2
      ],
      [
       9.5,
       57.8
      ],
      [
       10.0,
       58.3
      ]
     ]
    },
    {
     "id": 21,
     "cx": 0.6958,
     "cy": 0.5671,
     "w": 0.0609,
     "h": 0.1789,
     "mean": 27.4,
     "low_pct": 100.0,
     "cue": "glancing right",
     "n": 57,
     "series": [
      [
       1.5,
       18.4
      ],
      [
       2.0,
       19.8
      ],
      [
       2.5,
       21.5
      ],
      [
       3.0,
       28.9
      ],
      [
       3.5,
       28.2
      ],
      [
       4.0,
       21.4
      ],
      [
       4.5,
       18.8
      ],
      [
       5.0,
       20.0
      ],
      [
       5.5,
       25.1
      ],
      [
       6.0,
       34.2
      ],
      [
       6.5,
       38.9
      ],
      [
       7.0,
       42.6
      ],
      [
       7.5,
       46.3
      ]
     ]
    },
    {
     "id": 13,
     "cx": 0.3733,
     "cy": 0.3568,
     "w": 0.039,
     "h": 0.1022,
     "mean": 52.9,
     "low_pct": 6.0,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       1.5,
       53.1
      ],
      [
       2.0,
       52.8
      ],
      [
       2.5,
       52.7
      ],
      [
       3.0,
       52.3
      ],
      [
       3.5,
       51.3
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       49.8
      ],
      [
       5.0,
       50.2
      ],
      [
       5.5,
       53.4
      ],
      [
       6.0,
       53.4
      ],
      [
       6.5,
       52.3
      ],
      [
       7.0,
       51.5
      ],
      [
       7.5,
       51.8
      ],
      [
       8.0,
       52.7
      ],
      [
       8.5,
       54.8
      ],
      [
       9.0,
       56.7
      ],
      [
       9.5,
       58.2
      ],
      [
       10.0,
       59.3
      ]
     ]
    },
    {
     "id": 12,
     "cx": 0.3212,
     "cy": 0.4545,
     "w": 0.063,
     "h": 0.1804,
     "mean": 50.5,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       53.4
      ],
      [
       5.5,
       51.2
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.0
      ],
      [
       7.0,
       50.0
      ],
      [
       7.5,
       53.4
      ],
      [
       8.0,
       50.2
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       50.0
      ]
     ]
    },
    {
     "id": 15,
     "cx": 0.3084,
     "cy": 0.3131,
     "w": 0.033,
     "h": 0.091,
     "mean": 49.6,
     "low_pct": 25.3,
     "cue": "reading / taking notes",
     "n": 83,
     "series": [
      [
       1.5,
       52.5
      ],
      [
       2.0,
       51.5
      ],
      [
       2.5,
       50.8
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       48.6
      ],
      [
       4.5,
       44.8
      ],
      [
       5.0,
       43.9
      ],
      [
       5.5,
       44.1
      ],
      [
       6.0,
       48.8
      ],
      [
       6.5,
       52.3
      ],
      [
       7.0,
       52.3
      ],
      [
       7.5,
       50.8
      ],
      [
       8.0,
       50.0
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       51.5
      ],
      [
       9.5,
       52.5
      ],
      [
       10.0,
       53.7
      ]
     ]
    },
    {
     "id": 19,
     "cx": 0.4551,
     "cy": 0.3107,
     "w": 0.0258,
     "h": 0.0829,
     "mean": 42.8,
     "low_pct": 94.5,
     "cue": "reading / writing",
     "n": 73,
     "series": [
      [
       1.5,
       35.6
      ],
      [
       2.0,
       39.0
      ],
      [
       2.5,
       45.4
      ],
      [
       3.0,
       42.5
      ],
      [
       3.5,
       38.5
      ],
      [
       4.0,
       37.9
      ],
      [
       4.5,
       42.0
      ],
      [
       5.0,
       48.6
      ],
      [
       5.5,
       50.7
      ],
      [
       6.0,
       47.1
      ],
      [
       6.5,
       43.9
      ],
      [
       7.0,
       41.2
      ],
      [
       7.5,
       40.5
      ],
      [
       8.0,
       40.7
      ],
      [
       9.0,
       40.7
      ],
      [
       9.5,
       42.3
      ],
      [
       10.0,
       43.8
      ]
     ]
    },
    {
     "id": 17,
     "cx": 0.9628,
     "cy": 0.5435,
     "w": 0.0546,
     "h": 0.1551,
     "mean": 56.6,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       1.5,
       54.2
      ],
      [
       2.0,
       55.1
      ],
      [
       2.5,
       55.4
      ],
      [
       3.0,
       54.7
      ],
      [
       3.5,
       54.3
      ],
      [
       4.0,
       53.8
      ],
      [
       4.5,
       54.4
      ],
      [
       5.0,
       55.3
      ],
      [
       5.5,
       55.9
      ],
      [
       6.0,
       56.7
      ],
      [
       6.5,
       57.8
      ],
      [
       7.0,
       59.0
      ],
      [
       7.5,
       59.5
      ],
      [
       8.0,
       58.7
      ],
      [
       8.5,
       58.4
      ],
      [
       9.0,
       58.5
      ],
      [
       9.5,
       58.1
      ],
      [
       10.0,
       58.2
      ]
     ]
    },
    {
     "id": 18,
     "cx": 0.2699,
     "cy": 0.3644,
     "w": 0.0379,
     "h": 0.1053,
     "mean": 46.7,
     "low_pct": 41.0,
     "cue": "watching instructor",
     "n": 83,
     "series": [
      [
       1.5,
       30.0
      ],
      [
       2.0,
       30.0
      ],
      [
       2.5,
       51.7
      ],
      [
       3.0,
       51.0
      ],
      [
       3.5,
       51.3
      ],
      [
       4.0,
       51.2
      ],
      [
       4.5,
       52.0
      ],
      [
       5.0,
       52.6
      ],
      [
       5.5,
       48.5
      ],
      [
       6.0,
       39.2
      ],
      [
       6.5,
       36.9
      ],
      [
       7.0,
       37.5
      ],
      [
       7.5,
       43.9
      ],
      [
       8.0,
       49.3
      ],
      [
       8.5,
       50.6
      ],
      [
       9.0,
       51.5
      ],
      [
       9.5,
       51.3
      ],
      [
       10.0,
       52.1
      ]
     ]
    },
    {
     "id": 8,
     "cx": 0.4721,
     "cy": 0.3428,
     "w": 0.0379,
     "h": 0.0912,
     "mean": 37.5,
     "low_pct": 94.0,
     "cue": "watching instructor",
     "n": 83,
     "series": [
      [
       1.5,
       30.0
      ],
      [
       2.0,
       30.0
      ],
      [
       2.5,
       30.0
      ],
      [
       3.0,
       34.0
      ],
      [
       3.5,
       31.3
      ],
      [
       4.0,
       30.0
      ],
      [
       4.5,
       33.7
      ],
      [
       5.0,
       38.6
      ],
      [
       5.5,
       43.1
      ],
      [
       6.0,
       42.9
      ],
      [
       6.5,
       39.5
      ],
      [
       7.0,
       35.2
      ],
      [
       7.5,
       34.3
      ],
      [
       8.0,
       35.7
      ],
      [
       8.5,
       41.5
      ],
      [
       9.0,
       46.7
      ],
      [
       9.5,
       50.2
      ],
      [
       10.0,
       50.7
      ]
     ]
    },
    {
     "id": 20,
     "cx": 0.3748,
     "cy": 0.3206,
     "w": 0.0351,
     "h": 0.0839,
     "mean": 53.7,
     "low_pct": 1.2,
     "cue": "reading / writing",
     "n": 83,
     "series": [
      [
       1.5,
       52.2
      ],
      [
       2.0,
       52.4
      ],
      [
       2.5,
       52.7
      ],
      [
       3.0,
       53.4
      ],
      [
       3.5,
       53.0
      ],
      [
       4.0,
       52.1
      ],
      [
       4.5,
       52.9
      ],
      [
       5.0,
       54.1
      ],
      [
       5.5,
       55.0
      ],
      [
       6.0,
       55.6
      ],
      [
       6.5,
       54.5
      ],
      [
       7.0,
       52.2
      ],
      [
       7.5,
       50.6
      ],
      [
       8.0,
       50.5
      ],
      [
       8.5,
       53.2
      ],
      [
       9.0,
       56.4
      ],
      [
       9.5,
       58.0
      ],
      [
       10.0,
       59.4
      ]
     ]
    },
    {
     "id": 1,
     "cx": 0.5025,
     "cy": 0.687,
     "w": 0.0918,
     "h": 0.2619,
     "mean": 49.9,
     "low_pct": 9.8,
     "cue": "reading",
     "n": 82,
     "series": [
      [
       2.0,
       51.8
      ],
      [
       2.5,
       50.1
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       51.1
      ],
      [
       4.0,
       53.3
      ],
      [
       4.5,
       48.0
      ],
      [
       5.0,
       37.9
      ],
      [
       5.5,
       46.0
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.0
      ],
      [
       7.0,
       50.0
      ],
      [
       7.5,
       50.8
      ],
      [
       8.0,
       53.5
      ],
      [
       8.5,
       51.8
      ],
      [
       9.0,
       51.0
      ],
      [
       9.5,
       51.5
      ],
      [
       10.0,
       54.6
      ]
     ]
    },
    {
     "id": 5,
     "cx": 0.6439,
     "cy": 0.6005,
     "w": 0.0881,
     "h": 0.2231,
     "mean": 37.0,
     "low_pct": 81.7,
     "cue": "reading / writing",
     "n": 82,
     "series": [
      [
       2.0,
       45.5
      ],
      [
       2.5,
       39.5
      ],
      [
       3.0,
       30.0
      ],
      [
       3.5,
       41.4
      ],
      [
       4.0,
       34.0
      ],
      [
       4.5,
       29.4
      ],
      [
       5.0,
       25.6
      ],
      [
       5.5,
       24.2
      ],
      [
       6.0,
       24.7
      ],
      [
       6.5,
       26.1
      ],
      [
       7.0,
       29.6
      ],
      [
       7.5,
       35.5
      ],
      [
       8.0,
       41.9
      ],
      [
       8.5,
       45.7
      ],
      [
       9.0,
       51.0
      ],
      [
       9.5,
       57.9
      ],
      [
       10.0,
       61.3
      ]
     ]
    },
    {
     "id": 4,
     "cx": 0.8133,
     "cy": 0.415,
     "w": 0.0438,
     "h": 0.1236,
     "mean": 56.5,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "n": 82,
     "series": [
      [
       2.0,
       54.1
      ],
      [
       2.5,
       53.2
      ],
      [
       3.0,
       52.7
      ],
      [
       3.5,
       52.7
      ],
      [
       4.0,
       53.6
      ],
      [
       4.5,
       54.7
      ],
      [
       5.0,
       55.9
      ],
      [
       5.5,
       56.0
      ],
      [
       6.0,
       56.5
      ],
      [
       6.5,
       57.1
      ],
      [
       7.0,
       58.3
      ],
      [
       7.5,
       59.6
      ],
      [
       8.0,
       59.6
      ],
      [
       8.5,
       60.1
      ],
      [
       9.0,
       60.2
      ],
      [
       9.5,
       59.3
      ],
      [
       10.0,
       59.9
      ]
     ]
    },
    {
     "id": 2,
     "cx": 0.2842,
     "cy": 0.641,
     "w": 0.0902,
     "h": 0.2523,
     "mean": 29.4,
     "low_pct": 98.8,
     "cue": "using phone",
     "n": 82,
     "series": [
      [
       2.0,
       25.2
      ],
      [
       2.5,
       22.0
      ],
      [
       3.0,
       22.0
      ],
      [
       3.5,
       22.0
      ],
      [
       4.0,
       22.0
      ],
      [
       4.5,
       22.0
      ],
      [
       5.0,
       22.0
      ],
      [
       5.5,
       22.0
      ],
      [
       6.0,
       30.4
      ],
      [
       6.5,
       37.5
      ],
      [
       7.0,
       40.1
      ],
      [
       7.5,
       37.1
      ],
      [
       8.0,
       33.5
      ],
      [
       8.5,
       31.9
      ],
      [
       9.0,
       33.3
      ],
      [
       9.5,
       39.8
      ],
      [
       10.0,
       50.1
      ]
     ]
    },
    {
     "id": 6,
     "cx": 0.6607,
     "cy": 0.3494,
     "w": 0.0427,
     "h": 0.1117,
     "mean": 47.5,
     "low_pct": 35.4,
     "cue": "reading / writing",
     "n": 82,
     "series": [
      [
       2.0,
       58.7
      ],
      [
       2.5,
       57.9
      ],
      [
       3.0,
       57.2
      ],
      [
       3.5,
       56.8
      ],
      [
       4.0,
       57.0
      ],
      [
       4.5,
       57.5
      ],
      [
       5.0,
       53.3
      ],
      [
       5.5,
       40.7
      ],
      [
       6.0,
       37.3
      ],
      [
       6.5,
       32.5
      ],
      [
       7.0,
       31.9
      ],
      [
       7.5,
       31.4
      ],
      [
       8.0,
       34.2
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.6
      ],
      [
       10.0,
       53.1
      ]
     ]
    },
    {
     "id": 16,
     "cx": 0.7214,
     "cy": 0.4497,
     "w": 0.0573,
     "h": 0.125,
     "mean": 54.1,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "n": 82,
     "series": [
      [
       2.0,
       50.0
      ],
      [
       2.5,
       51.1
      ],
      [
       3.0,
       52.8
      ],
      [
       3.5,
       53.4
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       56.1
      ],
      [
       6.5,
       58.4
      ],
      [
       7.0,
       59.0
      ],
      [
       7.5,
       56.8
      ],
      [
       8.0,
       55.3
      ],
      [
       8.5,
       55.2
      ],
      [
       9.0,
       57.2
      ],
      [
       9.5,
       58.4
      ],
      [
       10.0,
       59.1
      ]
     ]
    },
    {
     "id": 3,
     "cx": 0.4647,
     "cy": 0.4278,
     "w": 0.0652,
     "h": 0.1614,
     "mean": 46.1,
     "low_pct": 51.2,
     "cue": "reading / writing",
     "n": 82,
     "series": [
      [
       2.0,
       49.6
      ],
      [
       2.5,
       50.1
      ],
      [
       3.0,
       46.5
      ],
      [
       3.5,
       45.2
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       49.0
      ],
      [
       6.5,
       40.8
      ],
      [
       7.0,
       35.0
      ],
      [
       7.5,
       33.5
      ],
      [
       8.0,
       35.2
      ],
      [
       8.5,
       42.9
      ],
      [
       9.0,
       50.6
      ],
      [
       9.5,
       55.3
      ],
      [
       10.0,
       57.0
      ]
     ]
    },
    {
     "id": 10,
     "cx": 0.1126,
     "cy": 0.6392,
     "w": 0.0866,
     "h": 0.2448,
     "mean": 49.3,
     "low_pct": 28.0,
     "cue": "reading / writing",
     "n": 82,
     "series": [
      [
       2.0,
       50.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       48.1
      ],
      [
       5.0,
       39.7
      ],
      [
       5.5,
       40.5
      ],
      [
       6.0,
       36.7
      ],
      [
       6.5,
       47.1
      ],
      [
       7.0,
       52.1
      ],
      [
       7.5,
       46.4
      ],
      [
       8.0,
       57.8
      ],
      [
       8.5,
       58.8
      ],
      [
       9.0,
       59.0
      ],
      [
       9.5,
       51.7
      ],
      [
       10.0,
       50.0
      ]
     ]
    }
   ],
   "class_mean": 46.85,
   "low_share": 37.65
  },
  {
   "key": "lecture_hall_video_project_3",
   "title": "Lecture hall (Video Project 3)",
   "room": "unlabelled",
   "labelled": false,
   "video": "media/lecture_hall_video_project_3.mp4",
   "poster": "media/lecture_hall_video_project_3.jpg",
   "frame": [
    1920,
    1080
   ],
   "duration": 22.5,
   "students_tracked": 52,
   "timeline": [
    [
     1.6,
     48.6,
     50.0,
     8
    ],
    [
     1.7,
     50.9,
     18.9,
     37
    ],
    [
     1.8,
     51.0,
     16.2,
     37
    ],
    [
     1.9,
     51.5,
     16.2,
     37
    ],
    [
     2.0,
     50.6,
     18.4,
     38
    ],
    [
     2.1,
     49.7,
     20.5,
     39
    ],
    [
     2.2,
     49.9,
     17.5,
     40
    ],
    [
     2.3,
     46.7,
     32.5,
     40
    ],
    [
     2.4,
     48.0,
     30.0,
     40
    ],
    [
     2.5,
     47.7,
     32.5,
     40
    ],
    [
     2.6,
     48.7,
     27.5,
     40
    ],
    [
     2.7,
     48.5,
     27.5,
     40
    ],
    [
     2.8,
     48.7,
     22.5,
     40
    ],
    [
     2.9,
     48.6,
     27.5,
     40
    ],
    [
     3.0,
     48.0,
     24.4,
     41
    ],
    [
     3.1,
     48.5,
     20.0,
     40
    ],
    [
     3.2,
     47.0,
     23.8,
     42
    ],
    [
     3.3,
     45.8,
     28.6,
     42
    ],
    [
     3.4,
     45.8,
     28.6,
     42
    ],
    [
     3.5,
     44.9,
     34.9,
     43
    ],
    [
     3.6,
     44.8,
     32.6,
     43
    ],
    [
     3.7,
     45.4,
     30.2,
     43
    ],
    [
     3.8,
     45.2,
     34.9,
     43
    ],
    [
     3.9,
     47.6,
     18.6,
     43
    ],
    [
     4.0,
     47.0,
     20.9,
     43
    ],
    [
     4.1,
     46.6,
     20.9,
     43
    ],
    [
     4.2,
     46.2,
     25.6,
     43
    ],
    [
     4.3,
     45.8,
     26.2,
     42
    ],
    [
     4.4,
     45.6,
     26.2,
     42
    ],
    [
     4.5,
     46.1,
     26.2,
     42
    ],
    [
     4.6,
     45.8,
     26.8,
     41
    ],
    [
     4.7,
     45.5,
     26.8,
     41
    ],
    [
     4.8,
     46.4,
     24.4,
     41
    ],
    [
     4.9,
     46.1,
     26.8,
     41
    ],
    [
     5.0,
     46.6,
     29.3,
     41
    ],
    [
     5.1,
     47.1,
     19.5,
     41
    ],
    [
     5.2,
     46.3,
     31.7,
     41
    ],
    [
     5.3,
     47.2,
     29.3,
     41
    ],
    [
     5.4,
     47.0,
     24.4,
     41
    ],
    [
     5.5,
     47.0,
     24.4,
     41
    ],
    [
     5.6,
     48.2,
     22.0,
     41
    ],
    [
     5.7,
     47.3,
     29.3,
     41
    ],
    [
     5.8,
     48.3,
     24.4,
     41
    ],
    [
     5.9,
     48.6,
     24.4,
     41
    ],
    [
     6.0,
     48.9,
     22.0,
     41
    ],
    [
     6.1,
     49.0,
     24.4,
     41
    ],
    [
     6.2,
     49.6,
     20.0,
     40
    ],
    [
     6.3,
     50.2,
     17.5,
     40
    ],
    [
     6.4,
     49.9,
     17.5,
     40
    ],
    [
     6.5,
     49.0,
     22.0,
     41
    ],
    [
     6.6,
     50.9,
     17.5,
     40
    ],
    [
     6.7,
     50.6,
     20.0,
     40
    ],
    [
     6.8,
     50.5,
     19.5,
     41
    ],
    [
     6.9,
     51.0,
     19.5,
     41
    ],
    [
     7.0,
     50.3,
     22.0,
     41
    ],
    [
     7.1,
     50.7,
     22.0,
     41
    ],
    [
     7.2,
     50.3,
     24.4,
     41
    ],
    [
     7.3,
     50.7,
     24.4,
     41
    ],
    [
     7.4,
     50.7,
     24.4,
     41
    ],
    [
     7.5,
     50.6,
     24.4,
     41
    ],
    [
     7.6,
     50.5,
     24.4,
     41
    ],
    [
     7.7,
     50.4,
     24.4,
     41
    ],
    [
     7.8,
     50.9,
     20.0,
     40
    ],
    [
     7.9,
     50.9,
     17.5,
     40
    ],
    [
     8.0,
     50.8,
     20.0,
     40
    ],
    [
     8.1,
     50.9,
     20.0,
     40
    ],
    [
     8.2,
     51.1,
     17.5,
     40
    ],
    [
     8.3,
     47.2,
     27.5,
     40
    ],
    [
     8.4,
     47.2,
     27.5,
     40
    ],
    [
     8.5,
     46.9,
     27.5,
     40
    ],
    [
     8.6,
     46.6,
     27.5,
     40
    ],
    [
     8.7,
     46.7,
     27.5,
     40
    ],
    [
     8.8,
     46.5,
     27.5,
     40
    ],
    [
     8.9,
     47.2,
     25.0,
     40
    ],
    [
     9.0,
     47.3,
     22.5,
     40
    ],
    [
     9.1,
     47.4,
     22.5,
     40
    ],
    [
     9.2,
     47.5,
     22.5,
     40
    ],
    [
     9.3,
     48.4,
     20.0,
     40
    ],
    [
     9.4,
     47.5,
     22.5,
     40
    ],
    [
     9.5,
     47.4,
     22.5,
     40
    ],
    [
     9.6,
     47.4,
     22.5,
     40
    ],
    [
     9.7,
     48.1,
     22.5,
     40
    ],
    [
     9.8,
     47.2,
     27.5,
     40
    ],
    [
     9.9,
     50.4,
     17.5,
     40
    ],
    [
     10.0,
     50.6,
     15.4,
     39
    ],
    [
     10.1,
     50.1,
     15.4,
     39
    ],
    [
     10.2,
     49.0,
     20.0,
     40
    ],
    [
     10.3,
     49.2,
     17.5,
     40
    ],
    [
     10.4,
     49.2,
     17.5,
     40
    ],
    [
     10.5,
     49.5,
     17.5,
     40
    ],
    [
     10.6,
     49.3,
     17.5,
     40
    ],
    [
     10.7,
     47.8,
     22.5,
     40
    ],
    [
     10.8,
     48.3,
     22.5,
     40
    ],
    [
     10.9,
     48.3,
     19.5,
     41
    ],
    [
     11.0,
     48.2,
     19.0,
     42
    ],
    [
     11.1,
     47.3,
     21.4,
     42
    ],
    [
     11.2,
     46.3,
     26.2,
     42
    ],
    [
     11.3,
     46.4,
     21.4,
     42
    ],
    [
     11.4,
     47.0,
     19.0,
     42
    ],
    [
     11.5,
     46.2,
     23.8,
     42
    ],
    [
     11.6,
     47.0,
     19.0,
     42
    ],
    [
     11.7,
     47.7,
     19.0,
     42
    ],
    [
     11.8,
     47.3,
     23.8,
     42
    ],
    [
     11.9,
     47.7,
     21.4,
     42
    ],
    [
     12.0,
     48.0,
     23.8,
     42
    ],
    [
     12.1,
     47.9,
     25.6,
     43
    ],
    [
     12.2,
     47.4,
     27.9,
     43
    ],
    [
     12.3,
     49.3,
     21.4,
     42
    ],
    [
     12.4,
     49.0,
     26.2,
     42
    ],
    [
     12.5,
     48.2,
     30.2,
     43
    ],
    [
     12.6,
     47.6,
     27.9,
     43
    ],
    [
     12.7,
     48.4,
     26.2,
     42
    ],
    [
     12.8,
     48.3,
     26.2,
     42
    ],
    [
     12.9,
     48.4,
     26.2,
     42
    ],
    [
     13.0,
     47.4,
     27.9,
     43
    ],
    [
     13.1,
     48.2,
     23.8,
     42
    ],
    [
     13.2,
     48.1,
     25.6,
     43
    ],
    [
     13.3,
     47.7,
     23.8,
     42
    ],
    [
     13.4,
     47.6,
     28.6,
     42
    ],
    [
     13.5,
     47.8,
     26.8,
     41
    ],
    [
     13.6,
     48.0,
     27.5,
     40
    ],
    [
     13.7,
     48.1,
     24.4,
     41
    ],
    [
     13.8,
     48.4,
     25.0,
     40
    ],
    [
     13.9,
     48.5,
     22.5,
     40
    ],
    [
     14.0,
     48.6,
     22.5,
     40
    ],
    [
     14.1,
     49.3,
     20.0,
     40
    ],
    [
     14.2,
     49.6,
     22.5,
     40
    ],
    [
     14.3,
     46.8,
     29.3,
     41
    ],
    [
     14.4,
     46.3,
     33.3,
     42
    ],
    [
     14.5,
     46.5,
     28.6,
     42
    ],
    [
     14.6,
     47.9,
     23.8,
     42
    ],
    [
     14.7,
     47.1,
     28.6,
     42
    ],
    [
     14.8,
     46.9,
     26.2,
     42
    ],
    [
     14.9,
     47.7,
     26.2,
     42
    ],
    [
     15.0,
     47.0,
     28.6,
     42
    ],
    [
     15.1,
     48.1,
     23.8,
     42
    ],
    [
     15.2,
     47.2,
     28.6,
     42
    ],
    [
     15.3,
     48.2,
     26.2,
     42
    ],
    [
     15.4,
     48.1,
     28.6,
     42
    ],
    [
     15.5,
     47.3,
     35.7,
     42
    ],
    [
     15.6,
     47.2,
     33.3,
     42
    ],
    [
     15.7,
     48.2,
     28.6,
     42
    ],
    [
     15.8,
     48.0,
     28.6,
     42
    ],
    [
     15.9,
     50.7,
     16.7,
     42
    ],
    [
     16.0,
     50.0,
     21.4,
     42
    ],
    [
     16.1,
     50.8,
     21.4,
     42
    ],
    [
     16.2,
     50.8,
     19.0,
     42
    ],
    [
     16.3,
     49.7,
     23.8,
     42
    ],
    [
     16.4,
     49.9,
     21.4,
     42
    ],
    [
     16.5,
     50.8,
     19.0,
     42
    ],
    [
     16.6,
     51.1,
     19.0,
     42
    ],
    [
     16.7,
     50.0,
     23.8,
     42
    ],
    [
     16.8,
     50.4,
     21.4,
     42
    ],
    [
     16.9,
     50.9,
     21.4,
     42
    ],
    [
     17.0,
     51.0,
     21.4,
     42
    ],
    [
     17.1,
     50.8,
     23.8,
     42
    ],
    [
     17.2,
     51.3,
     16.7,
     42
    ],
    [
     17.3,
     50.9,
     19.0,
     42
    ],
    [
     17.4,
     51.6,
     16.7,
     42
    ],
    [
     17.5,
     50.7,
     19.0,
     42
    ],
    [
     17.6,
     50.7,
     21.4,
     42
    ],
    [
     17.7,
     51.5,
     19.0,
     42
    ],
    [
     17.8,
     51.5,
     19.0,
     42
    ],
    [
     17.9,
     51.5,
     14.3,
     42
    ],
    [
     18.0,
     51.7,
     14.3,
     42
    ],
    [
     18.1,
     51.8,
     14.3,
     42
    ],
    [
     18.2,
     51.2,
     11.9,
     42
    ],
    [
     18.3,
     52.5,
     9.5,
     42
    ],
    [
     18.4,
     52.5,
     9.5,
     42
    ],
    [
     18.5,
     50.5,
     17.1,
     41
    ],
    [
     18.6,
     50.2,
     17.1,
     41
    ],
    [
     18.7,
     50.3,
     17.1,
     41
    ],
    [
     18.8,
     51.2,
     17.1,
     41
    ],
    [
     18.9,
     49.6,
     23.8,
     42
    ],
    [
     19.0,
     49.6,
     23.8,
     42
    ],
    [
     19.1,
     49.7,
     23.8,
     42
    ],
    [
     19.2,
     49.6,
     26.2,
     42
    ],
    [
     19.3,
     50.1,
     23.8,
     42
    ],
    [
     19.4,
     50.1,
     23.8,
     42
    ],
    [
     19.5,
     50.0,
     26.2,
     42
    ],
    [
     19.6,
     50.1,
     23.8,
     42
    ],
    [
     19.7,
     49.7,
     28.6,
     42
    ],
    [
     19.8,
     50.2,
     26.2,
     42
    ],
    [
     19.9,
     50.7,
     23.8,
     42
    ],
    [
     20.0,
     50.3,
     23.8,
     42
    ],
    [
     20.1,
     50.7,
     19.0,
     42
    ],
    [
     20.2,
     50.8,
     21.4,
     42
    ],
    [
     20.3,
     47.5,
     28.6,
     42
    ],
    [
     20.4,
     47.8,
     26.2,
     42
    ],
    [
     20.5,
     47.4,
     28.6,
     42
    ],
    [
     20.6,
     47.6,
     28.6,
     42
    ],
    [
     20.7,
     48.0,
     28.6,
     42
    ],
    [
     20.8,
     47.4,
     26.2,
     42
    ],
    [
     20.9,
     47.4,
     28.6,
     42
    ],
    [
     21.0,
     47.2,
     28.6,
     42
    ],
    [
     21.1,
     48.0,
     26.2,
     42
    ],
    [
     21.2,
     48.0,
     21.4,
     42
    ],
    [
     21.3,
     47.1,
     29.3,
     41
    ],
    [
     21.4,
     47.5,
     25.0,
     40
    ],
    [
     21.5,
     47.4,
     29.3,
     41
    ],
    [
     21.6,
     46.6,
     31.7,
     41
    ],
    [
     21.7,
     46.8,
     31.7,
     41
    ],
    [
     21.8,
     46.7,
     31.7,
     41
    ],
    [
     21.9,
     49.7,
     19.5,
     41
    ],
    [
     22.0,
     50.0,
     20.0,
     40
    ],
    [
     22.1,
     49.7,
     22.5,
     40
    ],
    [
     22.2,
     50.3,
     20.5,
     39
    ],
    [
     22.3,
     51.6,
     15.4,
     39
    ],
    [
     22.4,
     51.8,
     13.2,
     38
    ],
    [
     22.5,
     50.9,
     18.4,
     38
    ]
   ],
   "students": [
    {
     "id": 42,
     "n": 2,
     "mean": 13.6,
     "low_pct": 100.0,
     "cue": "oriented to lesson",
     "alerts": 0,
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 16,
     "n": 13,
     "mean": 16.0,
     "low_pct": 100.0,
     "cue": "reading / writing",
     "alerts": 0,
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 48,
     "n": 19,
     "mean": 21.9,
     "low_pct": 100.0,
     "cue": "oriented to lesson",
     "alerts": 0,
     "daisee": 84.7,
     "daisee_aux": [
      2.44,
      0.33,
      0.24,
      0.26
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 45,
     "n": 25,
     "mean": 22.5,
     "low_pct": 100.0,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 92.5,
     "daisee_aux": [
      2.55,
      0.51,
      0.37,
      0.02
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 19,
     "n": 116,
     "mean": 27.4,
     "low_pct": 80.2,
     "cue": "glancing right",
     "alerts": 2,
     "daisee": 83.1,
     "daisee_aux": [
      2.41,
      0.27,
      0.05,
      0.32
     ],
     "scb": "reading",
     "scb_p": 0.25
    },
    {
     "id": 44,
     "n": 17,
     "mean": 28.2,
     "low_pct": 100.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 66.8,
     "daisee_aux": [
      2.22,
      0.67,
      0.34,
      0.36
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 51,
     "n": 69,
     "mean": 28.2,
     "low_pct": 100.0,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 76.2,
     "daisee_aux": [
      2.37,
      0.75,
      0.21,
      0.17
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 23,
     "n": 45,
     "mean": 34.5,
     "low_pct": 88.9,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 67.8,
     "daisee_aux": [
      2.28,
      0.62,
      0.2,
      0.28
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 33,
     "n": 209,
     "mean": 35.6,
     "low_pct": 48.8,
     "cue": "reading",
     "alerts": 3,
     "daisee": 66.5,
     "daisee_aux": [
      2.05,
      0.65,
      -0.03,
      0.2
     ],
     "scb": "reading",
     "scb_p": 1.0
    },
    {
     "id": 46,
     "n": 159,
     "mean": 37.7,
     "low_pct": 89.9,
     "cue": "reading / writing",
     "alerts": 1,
     "daisee": 76.4,
     "daisee_aux": [
      2.43,
      0.45,
      0.28,
      0.23
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 52,
     "n": 70,
     "mean": 38.0,
     "low_pct": 94.3,
     "cue": "reading / taking notes",
     "alerts": 1,
     "daisee": 68.0,
     "daisee_aux": [
      2.23,
      0.53,
      0.14,
      0.25
     ],
     "scb": "reading",
     "scb_p": 0.21
    },
    {
     "id": 28,
     "n": 201,
     "mean": 40.1,
     "low_pct": 67.2,
     "cue": "reading / taking notes",
     "alerts": 2,
     "daisee": 71.2,
     "daisee_aux": [
      2.3,
      0.45,
      0.15,
      0.21
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 2,
     "n": 209,
     "mean": 40.4,
     "low_pct": 84.2,
     "cue": "reading / writing",
     "alerts": 2,
     "daisee": 68.0,
     "daisee_aux": [
      2.28,
      0.6,
      0.23,
      0.32
     ],
     "scb": "reading",
     "scb_p": 0.17
    },
    {
     "id": 41,
     "n": 140,
     "mean": 41.1,
     "low_pct": 33.6,
     "cue": "reading",
     "alerts": 1,
     "daisee": 87.7,
     "daisee_aux": [
      2.57,
      0.58,
      0.11,
      0.14
     ],
     "scb": "reading",
     "scb_p": 0.72
    },
    {
     "id": 36,
     "n": 204,
     "mean": 42.4,
     "low_pct": 92.6,
     "cue": "reading / taking notes",
     "alerts": 1,
     "daisee": 67.4,
     "daisee_aux": [
      2.33,
      0.56,
      0.17,
      0.34
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 38,
     "n": 206,
     "mean": 42.5,
     "low_pct": 25.2,
     "cue": "reading / taking notes",
     "alerts": 2,
     "daisee": 82.8,
     "daisee_aux": [
      2.48,
      0.27,
      0.02,
      0.21
     ],
     "scb": "reading",
     "scb_p": 0.56
    },
    {
     "id": 32,
     "n": 209,
     "mean": 43.6,
     "low_pct": 72.2,
     "cue": "slouched, watching lesson",
     "alerts": 1,
     "daisee": 72.3,
     "daisee_aux": [
      2.3,
      0.38,
      0.06,
      0.28
     ],
     "scb": "leaning on table",
     "scb_p": 0.37
    },
    {
     "id": 39,
     "n": 27,
     "mean": 45.2,
     "low_pct": 88.9,
     "cue": "slouched, watching lesson",
     "alerts": 0,
     "daisee": 69.8,
     "daisee_aux": [
      2.34,
      0.39,
      0.12,
      0.34
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 37,
     "n": 210,
     "mean": 48.2,
     "low_pct": 62.9,
     "cue": "glancing right",
     "alerts": 1,
     "daisee": 85.3,
     "daisee_aux": [
      2.37,
      0.18,
      0.04,
      0.31
     ],
     "scb": "reading",
     "scb_p": 0.33
    },
    {
     "id": 55,
     "n": 11,
     "mean": 48.3,
     "low_pct": 72.7,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 75.6,
     "daisee_aux": [
      2.36,
      0.31,
      0.14,
      0.36
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 40,
     "n": 204,
     "mean": 48.4,
     "low_pct": 60.8,
     "cue": "reading / taking notes",
     "alerts": 2,
     "daisee": 69.0,
     "daisee_aux": [
      2.39,
      0.77,
      0.24,
      0.31
     ],
     "scb": "reading",
     "scb_p": 0.19
    },
    {
     "id": 26,
     "n": 209,
     "mean": 49.1,
     "low_pct": 7.7,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 75.2,
     "daisee_aux": [
      2.27,
      0.31,
      0.26,
      0.27
     ],
     "scb": "reading",
     "scb_p": 0.61
    },
    {
     "id": 35,
     "n": 209,
     "mean": 49.3,
     "low_pct": 56.9,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 68.8,
     "daisee_aux": [
      2.4,
      0.82,
      0.08,
      0.27
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 21,
     "n": 209,
     "mean": 49.9,
     "low_pct": 5.7,
     "cue": "reading",
     "alerts": 0,
     "daisee": 73.1,
     "daisee_aux": [
      2.32,
      0.71,
      -0.05,
      0.09
     ],
     "scb": "reading",
     "scb_p": 0.79
    },
    {
     "id": 9,
     "n": 209,
     "mean": 50.0,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 91.0,
     "daisee_aux": [
      2.8,
      0.56,
      0.05,
      0.06
     ],
     "scb": "reading",
     "scb_p": 0.97
    },
    {
     "id": 24,
     "n": 209,
     "mean": 50.0,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 70.0,
     "daisee_aux": [
      2.2,
      0.49,
      -0.07,
      0.14
     ],
     "scb": "reading",
     "scb_p": 0.61
    },
    {
     "id": 11,
     "n": 209,
     "mean": 50.1,
     "low_pct": 50.7,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.0,
     "daisee_aux": [
      2.17,
      0.58,
      0.16,
      0.29
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 13,
     "n": 209,
     "mean": 50.1,
     "low_pct": 0.0,
     "cue": "reading",
     "alerts": 0,
     "daisee": 85.1,
     "daisee_aux": [
      2.6,
      0.72,
      0.09,
      0.1
     ],
     "scb": "reading",
     "scb_p": 0.76
    },
    {
     "id": 18,
     "n": 209,
     "mean": 50.2,
     "low_pct": 45.5,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 66.9,
     "daisee_aux": [
      2.39,
      0.81,
      0.12,
      0.26
     ],
     "scb": "hand-raising",
     "scb_p": 0.17
    },
    {
     "id": 27,
     "n": 206,
     "mean": 50.6,
     "low_pct": 1.5,
     "cue": "reading",
     "alerts": 0,
     "daisee": 86.8,
     "daisee_aux": [
      2.67,
      0.66,
      0.03,
      0.06
     ],
     "scb": "reading",
     "scb_p": 0.99
    },
    {
     "id": 25,
     "n": 209,
     "mean": 50.8,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.7,
     "daisee_aux": [
      2.25,
      0.54,
      -0.03,
      0.18
     ],
     "scb": "reading",
     "scb_p": 0.58
    },
    {
     "id": 49,
     "n": 117,
     "mean": 50.8,
     "low_pct": 3.4,
     "cue": "reading / taking notes",
     "alerts": 1,
     "daisee": 67.6,
     "daisee_aux": [
      2.18,
      0.71,
      0.04,
      0.18
     ],
     "scb": "reading",
     "scb_p": 0.57
    },
    {
     "id": 10,
     "n": 209,
     "mean": 51.1,
     "low_pct": 0.0,
     "cue": "reading",
     "alerts": 0,
     "daisee": 85.0,
     "daisee_aux": [
      2.44,
      0.78,
      0.09,
      0.1
     ],
     "scb": "reading",
     "scb_p": 0.8
    },
    {
     "id": 34,
     "n": 208,
     "mean": 51.2,
     "low_pct": 0.0,
     "cue": "reading",
     "alerts": 0,
     "daisee": 91.3,
     "daisee_aux": [
      2.5,
      0.54,
      0.03,
      0.15
     ],
     "scb": "reading",
     "scb_p": 0.83
    },
    {
     "id": 31,
     "n": 193,
     "mean": 51.4,
     "low_pct": 0.0,
     "cue": "reading",
     "alerts": 0,
     "daisee": 95.5,
     "daisee_aux": [
      2.71,
      0.57,
      0.06,
      0.08
     ],
     "scb": "reading",
     "scb_p": 1.0
    },
    {
     "id": 4,
     "n": 209,
     "mean": 51.7,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.2,
     "daisee_aux": [
      2.45,
      0.29,
      0.0,
      0.18
     ],
     "scb": "reading",
     "scb_p": 0.58
    },
    {
     "id": 15,
     "n": 209,
     "mean": 51.8,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 66.8,
     "daisee_aux": [
      2.29,
      0.35,
      0.09,
      0.2
     ],
     "scb": "reading",
     "scb_p": 0.58
    },
    {
     "id": 20,
     "n": 209,
     "mean": 52.0,
     "low_pct": 0.5,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 98.2,
     "daisee_aux": [
      3.08,
      0.78,
      0.15,
      0.13
     ],
     "scb": "reading",
     "scb_p": 0.58
    },
    {
     "id": 1,
     "n": 209,
     "mean": 52.2,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 69.2,
     "daisee_aux": [
      2.51,
      0.22,
      -0.04,
      0.17
     ],
     "scb": "reading",
     "scb_p": 0.58
    },
    {
     "id": 50,
     "n": 105,
     "mean": 52.3,
     "low_pct": 10.5,
     "cue": "reading",
     "alerts": 0,
     "daisee": 69.0,
     "daisee_aux": [
      2.18,
      0.42,
      0.01,
      0.22
     ],
     "scb": "reading",
     "scb_p": 0.53
    },
    {
     "id": 29,
     "n": 209,
     "mean": 52.5,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 68.8,
     "daisee_aux": [
      2.42,
      0.39,
      -0.02,
      0.17
     ],
     "scb": "reading",
     "scb_p": 0.58
    },
    {
     "id": 17,
     "n": 204,
     "mean": 52.9,
     "low_pct": 4.9,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 70.8,
     "daisee_aux": [
      2.16,
      0.7,
      -0.03,
      0.19
     ],
     "scb": "reading",
     "scb_p": 0.55
    },
    {
     "id": 5,
     "n": 209,
     "mean": 53.0,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 69.5,
     "daisee_aux": [
      2.2,
      0.79,
      -0.01,
      0.12
     ],
     "scb": "reading",
     "scb_p": 0.62
    },
    {
     "id": 22,
     "n": 209,
     "mean": 53.0,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 67.8,
     "daisee_aux": [
      2.37,
      0.42,
      -0.01,
      0.18
     ],
     "scb": "reading",
     "scb_p": 0.58
    },
    {
     "id": 7,
     "n": 209,
     "mean": 53.4,
     "low_pct": 0.0,
     "cue": "reading",
     "alerts": 0,
     "daisee": 99.3,
     "daisee_aux": [
      2.52,
      -0.01,
      -0.21,
      0.11
     ],
     "scb": "reading",
     "scb_p": 1.0
    },
    {
     "id": 14,
     "n": 209,
     "mean": 53.5,
     "low_pct": 9.1,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 68.1,
     "daisee_aux": [
      2.48,
      0.33,
      0.03,
      0.21
     ],
     "scb": "reading",
     "scb_p": 0.37
    },
    {
     "id": 12,
     "n": 209,
     "mean": 53.6,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "alerts": 0,
     "daisee": 68.0,
     "daisee_aux": [
      2.41,
      0.1,
      -0.08,
      0.16
     ],
     "scb": "reading",
     "scb_p": 0.59
    },
    {
     "id": 30,
     "n": 210,
     "mean": 53.7,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 98.3,
     "daisee_aux": [
      2.76,
      0.69,
      0.14,
      0.13
     ],
     "scb": "reading",
     "scb_p": 0.93
    },
    {
     "id": 8,
     "n": 209,
     "mean": 53.9,
     "low_pct": 0.0,
     "cue": "reading",
     "alerts": 0,
     "daisee": 93.8,
     "daisee_aux": [
      2.91,
      0.49,
      0.05,
      0.07
     ],
     "scb": "reading",
     "scb_p": 1.0
    },
    {
     "id": 6,
     "n": 209,
     "mean": 54.0,
     "low_pct": 0.0,
     "cue": "reading",
     "alerts": 0,
     "daisee": 96.3,
     "daisee_aux": [
      2.74,
      0.43,
      0.05,
      0.1
     ],
     "scb": "reading",
     "scb_p": 0.95
    },
    {
     "id": 3,
     "n": 209,
     "mean": 54.7,
     "low_pct": 0.0,
     "cue": "reading",
     "alerts": 0,
     "daisee": 96.6,
     "daisee_aux": [
      2.64,
      -0.07,
      -0.12,
      0.13
     ],
     "scb": "reading",
     "scb_p": 0.85
    },
    {
     "id": 43,
     "n": 197,
     "mean": 55.9,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "alerts": 0,
     "daisee": 68.9,
     "daisee_aux": [
      2.33,
      0.24,
      0.11,
      0.22
     ],
     "scb": "reading",
     "scb_p": 0.58
    }
   ],
   "alerts": [
    {
     "t": 2.9,
     "student": 33,
     "kind": "sudden",
     "score": 26.1,
     "cue": "reading",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 3.2,
     "student": 37,
     "kind": "sudden",
     "score": 48.4,
     "cue": "glancing right",
     "recommendation": "Direct an inclusive, low-stakes retrieval question toward that sector of the room, or use a vocal inflection change to refocus eyes on the visual aid.",
     "suggestion": "Wait one brief observation cycle; isolated look-aways are normal cognitive breathers."
    },
    {
     "t": 3.4,
     "student": 19,
     "kind": "sudden",
     "score": 29.7,
     "cue": "glancing right",
     "recommendation": "Direct an inclusive, low-stakes retrieval question toward that sector of the room, or use a vocal inflection change to refocus eyes on the visual aid.",
     "suggestion": "Circulate toward that side of the room and re-anchor visual engagement."
    },
    {
     "t": 3.8,
     "student": 23,
     "kind": "sudden",
     "score": 29.8,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 4.2,
     "student": 2,
     "kind": "sustained",
     "score": 30.9,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 4.6,
     "student": 28,
     "kind": "sustained",
     "score": 23.7,
     "cue": "reading / taking notes",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 6.8,
     "student": 32,
     "kind": "sustained",
     "score": 31.4,
     "cue": "slouched, watching lesson",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 7.0,
     "student": 45,
     "kind": "sustained",
     "score": 22.3,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 10.2,
     "student": 46,
     "kind": "sustained",
     "score": 19.3,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 11.4,
     "student": 19,
     "kind": "sudden",
     "score": 11.8,
     "cue": "glancing right",
     "recommendation": "Direct an inclusive, low-stakes retrieval question toward that sector of the room, or use a vocal inflection change to refocus eyes on the visual aid.",
     "suggestion": "Circulate toward that side of the room and re-anchor visual engagement."
    },
    {
     "t": 11.4,
     "student": 49,
     "kind": "sudden",
     "score": 28.7,
     "cue": "reading / taking notes",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 11.7,
     "student": 33,
     "kind": "sudden",
     "score": 25.0,
     "cue": "reading",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 12.2,
     "student": 40,
     "kind": "sudden",
     "score": 34.3,
     "cue": "reading / taking notes",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Wait one more cycle; brief dips are common."
    },
    {
     "t": 12.3,
     "student": 41,
     "kind": "sudden",
     "score": 29.2,
     "cue": "reading",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 12.3,
     "student": 2,
     "kind": "sustained",
     "score": 39.0,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 12.6,
     "student": 28,
     "kind": "sustained",
     "score": 37.9,
     "cue": "reading / taking notes",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 13.8,
     "student": 38,
     "kind": "sudden",
     "score": 28.4,
     "cue": "reading / taking notes",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 16.8,
     "student": 51,
     "kind": "sustained",
     "score": 30.9,
     "cue": "reading / writing",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 19.7,
     "student": 33,
     "kind": "sudden",
     "score": 20.9,
     "cue": "reading",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 19.9,
     "student": 36,
     "kind": "sudden",
     "score": 33.7,
     "cue": "reading / taking notes",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Wait one more cycle; brief dips are common."
    },
    {
     "t": 21.3,
     "student": 40,
     "kind": "sudden",
     "score": 47.7,
     "cue": "reading / taking notes",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Wait one more cycle; brief dips are common."
    },
    {
     "t": 21.8,
     "student": 38,
     "kind": "sustained",
     "score": 17.7,
     "cue": "reading / taking notes",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    },
    {
     "t": 21.9,
     "student": 52,
     "kind": "sustained",
     "score": 38.7,
     "cue": "reading / taking notes",
     "recommendation": "Pause for a 15-second retrieval prompt aimed at the whole class, then cold-call a nearby volunteer rather than the flagged seat.",
     "suggestion": "Circulate toward that side of the room and use a quiet check-in."
    }
   ],
   "seats": [
    {
     "id": 30,
     "cx": 0.5235,
     "cy": 0.253,
     "w": 0.0334,
     "h": 0.0657,
     "mean": 53.7,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "n": 210,
     "series": [
      [
       1.5,
       55.2
      ],
      [
       2.0,
       56.4
      ],
      [
       2.5,
       50.5
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.2
      ],
      [
       4.5,
       50.1
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.0
      ],
      [
       7.0,
       50.2
      ],
      [
       7.5,
       50.9
      ],
      [
       8.0,
       51.7
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       51.0
      ],
      [
       9.5,
       52.0
      ],
      [
       10.0,
       53.9
      ],
      [
       10.5,
       53.0
      ],
      [
       11.0,
       51.1
      ],
      [
       11.5,
       52.8
      ],
      [
       12.0,
       53.2
      ],
      [
       12.5,
       55.8
      ],
      [
       13.0,
       56.4
      ],
      [
       13.5,
       56.9
      ],
      [
       14.0,
       56.9
      ],
      [
       14.5,
       51.7
      ],
      [
       15.0,
       52.4
      ],
      [
       15.5,
       51.6
      ],
      [
       16.0,
       54.3
      ],
      [
       16.5,
       56.5
      ],
      [
       17.0,
       58.2
      ],
      [
       17.5,
       61.6
      ],
      [
       18.0,
       61.5
      ],
      [
       18.5,
       59.9
      ],
      [
       19.0,
       57.9
      ],
      [
       19.5,
       58.4
      ],
      [
       20.0,
       58.9
      ],
      [
       20.5,
       53.3
      ],
      [
       21.0,
       52.3
      ],
      [
       21.5,
       50.9
      ],
      [
       22.0,
       56.2
      ],
      [
       22.5,
       58.5
      ]
     ]
    },
    {
     "id": 37,
     "cx": 0.3783,
     "cy": 0.2594,
     "w": 0.0266,
     "h": 0.0452,
     "mean": 48.2,
     "low_pct": 62.9,
     "cue": "glancing right",
     "n": 210,
     "series": [
      [
       1.5,
       69.3
      ],
      [
       2.0,
       67.8
      ],
      [
       2.5,
       55.3
      ],
      [
       3.0,
       48.3
      ],
      [
       3.5,
       40.1
      ],
      [
       4.0,
       38.1
      ],
      [
       4.5,
       38.6
      ],
      [
       5.0,
       41.3
      ],
      [
       5.5,
       45.0
      ],
      [
       6.0,
       45.1
      ],
      [
       6.5,
       46.4
      ],
      [
       7.0,
       47.7
      ],
      [
       7.5,
       48.8
      ],
      [
       8.0,
       49.5
      ],
      [
       8.5,
       46.7
      ],
      [
       9.0,
       46.9
      ],
      [
       9.5,
       48.0
      ],
      [
       10.0,
       51.6
      ],
      [
       10.5,
       52.9
      ],
      [
       11.0,
       51.7
      ],
      [
       11.5,
       50.0
      ],
      [
       12.0,
       39.5
      ],
      [
       12.5,
       38.1
      ],
      [
       13.0,
       43.9
      ],
      [
       13.5,
       45.0
      ],
      [
       14.0,
       48.3
      ],
      [
       14.5,
       45.3
      ],
      [
       15.0,
       46.0
      ],
      [
       15.5,
       47.2
      ],
      [
       16.0,
       51.7
      ],
      [
       16.5,
       53.3
      ],
      [
       17.0,
       52.1
      ],
      [
       17.5,
       50.4
      ],
      [
       18.0,
       51.4
      ],
      [
       18.5,
       52.2
      ],
      [
       19.0,
       55.6
      ],
      [
       19.5,
       54.1
      ],
      [
       20.0,
       48.4
      ],
      [
       20.5,
       45.3
      ],
      [
       21.0,
       45.2
      ],
      [
       21.5,
       47.2
      ],
      [
       22.0,
       47.9
      ],
      [
       22.5,
       47.8
      ]
     ]
    },
    {
     "id": 34,
     "cx": 0.5764,
     "cy": 0.2763,
     "w": 0.0354,
     "h": 0.0774,
     "mean": 51.2,
     "low_pct": 0.0,
     "cue": "reading",
     "n": 208,
     "series": [
      [
       1.5,
       50.1
      ],
      [
       2.0,
       53.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       51.0
      ],
      [
       3.5,
       50.4
      ],
      [
       4.0,
       50.2
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.2
      ],
      [
       7.0,
       52.8
      ],
      [
       7.5,
       53.7
      ],
      [
       8.0,
       53.7
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       50.9
      ],
      [
       10.5,
       51.3
      ],
      [
       11.0,
       50.0
      ],
      [
       11.5,
       50.0
      ],
      [
       12.0,
       50.0
      ],
      [
       12.5,
       50.0
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.0
      ],
      [
       14.0,
       50.0
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       50.0
      ],
      [
       16.5,
       50.0
      ],
      [
       17.0,
       50.0
      ],
      [
       17.5,
       50.8
      ],
      [
       18.0,
       54.6
      ],
      [
       18.5,
       60.2
      ],
      [
       19.0,
       61.6
      ],
      [
       19.5,
       55.9
      ],
      [
       20.0,
       50.0
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       50.0
      ],
      [
       22.5,
       50.0
      ]
     ]
    },
    {
     "id": 36,
     "cx": 0.6215,
     "cy": 0.2328,
     "w": 0.0235,
     "h": 0.0736,
     "mean": 42.4,
     "low_pct": 92.6,
     "cue": "reading / taking notes",
     "n": 204,
     "series": [
      [
       1.5,
       44.6
      ],
      [
       2.0,
       43.2
      ],
      [
       2.5,
       40.9
      ],
      [
       3.0,
       43.4
      ],
      [
       3.5,
       44.5
      ],
      [
       4.0,
       48.5
      ],
      [
       4.5,
       47.6
      ],
      [
       5.0,
       49.5
      ],
      [
       5.5,
       48.7
      ],
      [
       6.0,
       46.8
      ],
      [
       6.5,
       45.1
      ],
      [
       7.0,
       45.2
      ],
      [
       7.5,
       46.8
      ],
      [
       8.0,
       35.9
      ],
      [
       8.5,
       30.4
      ],
      [
       9.0,
       28.2
      ],
      [
       9.5,
       30.5
      ],
      [
       10.0,
       36.6
      ],
      [
       10.5,
       42.7
      ],
      [
       11.0,
       47.5
      ],
      [
       11.5,
       47.9
      ],
      [
       12.0,
       47.5
      ],
      [
       12.5,
       49.3
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       47.5
      ],
      [
       14.0,
       45.7
      ],
      [
       14.5,
       42.5
      ],
      [
       15.0,
       42.2
      ],
      [
       15.5,
       42.8
      ],
      [
       16.0,
       45.1
      ],
      [
       16.5,
       44.4
      ],
      [
       17.0,
       45.9
      ],
      [
       17.5,
       46.9
      ],
      [
       18.0,
       52.0
      ],
      [
       18.5,
       55.0
      ],
      [
       19.0,
       43.8
      ],
      [
       19.5,
       34.6
      ],
      [
       20.0,
       30.7
      ],
      [
       20.5,
       28.4
      ],
      [
       21.0,
       30.6
      ],
      [
       21.5,
       28.2
      ],
      [
       22.0,
       25.3
      ]
     ]
    },
    {
     "id": 38,
     "cx": 0.6404,
     "cy": 0.3073,
     "w": 0.0402,
     "h": 0.0869,
     "mean": 42.5,
     "low_pct": 25.2,
     "cue": "reading / taking notes",
     "n": 206,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.0
      ],
      [
       7.0,
       50.0
      ],
      [
       7.5,
       50.0
      ],
      [
       8.0,
       50.0
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       51.8
      ],
      [
       10.5,
       52.8
      ],
      [
       11.0,
       50.0
      ],
      [
       11.5,
       50.0
      ],
      [
       12.0,
       50.0
      ],
      [
       12.5,
       50.0
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       33.1
      ],
      [
       14.0,
       33.0
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       27.9
      ],
      [
       15.5,
       36.6
      ],
      [
       16.0,
       50.0
      ],
      [
       16.5,
       50.0
      ],
      [
       17.0,
       50.0
      ],
      [
       17.5,
       50.0
      ],
      [
       18.0,
       50.0
      ],
      [
       18.5,
       32.9
      ],
      [
       19.0,
       15.4
      ],
      [
       19.5,
       14.2
      ],
      [
       20.0,
       14.6
      ],
      [
       20.5,
       18.5
      ],
      [
       21.0,
       20.2
      ],
      [
       21.5,
       17.7
      ],
      [
       22.0,
       18.2
      ]
     ]
    },
    {
     "id": 40,
     "cx": 0.5714,
     "cy": 0.2431,
     "w": 0.0251,
     "h": 0.0797,
     "mean": 48.4,
     "low_pct": 60.8,
     "cue": "reading / taking notes",
     "n": 204,
     "series": [
      [
       1.5,
       46.0
      ],
      [
       2.0,
       50.2
      ],
      [
       2.5,
       47.3
      ],
      [
       3.0,
       49.7
      ],
      [
       3.5,
       49.8
      ],
      [
       4.0,
       50.3
      ],
      [
       4.5,
       47.2
      ],
      [
       5.0,
       46.3
      ],
      [
       5.5,
       46.7
      ],
      [
       6.0,
       48.5
      ],
      [
       6.5,
       51.5
      ],
      [
       7.0,
       56.0
      ],
      [
       7.5,
       55.1
      ],
      [
       8.0,
       52.6
      ],
      [
       8.5,
       46.7
      ],
      [
       9.0,
       47.7
      ],
      [
       9.5,
       49.0
      ],
      [
       10.0,
       53.9
      ],
      [
       10.5,
       54.5
      ],
      [
       11.0,
       44.2
      ],
      [
       11.5,
       36.2
      ],
      [
       12.0,
       34.0
      ],
      [
       12.5,
       39.9
      ],
      [
       13.0,
       35.1
      ],
      [
       13.5,
       41.5
      ],
      [
       14.5,
       32.1
      ],
      [
       15.0,
       34.8
      ],
      [
       15.5,
       40.2
      ],
      [
       16.0,
       47.9
      ],
      [
       16.5,
       47.6
      ],
      [
       17.0,
       49.5
      ],
      [
       17.5,
       53.1
      ],
      [
       18.0,
       57.4
      ],
      [
       18.5,
       64.4
      ],
      [
       19.0,
       65.2
      ],
      [
       19.5,
       68.3
      ],
      [
       20.0,
       61.1
      ],
      [
       20.5,
       50.4
      ],
      [
       21.0,
       48.0
      ],
      [
       21.5,
       45.0
      ],
      [
       22.0,
       39.5
      ],
      [
       22.5,
       45.5
      ]
     ]
    },
    {
     "id": 41,
     "cx": 0.6774,
     "cy": 0.2904,
     "w": 0.0256,
     "h": 0.0723,
     "mean": 41.1,
     "low_pct": 33.6,
     "cue": "reading",
     "n": 140,
     "series": [
      [
       1.5,
       32.9
      ],
      [
       2.0,
       29.2
      ],
      [
       2.5,
       45.2
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.0
      ],
      [
       7.0,
       50.0
      ],
      [
       7.5,
       50.0
      ],
      [
       8.0,
       50.0
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       50.0
      ],
      [
       10.5,
       45.0
      ],
      [
       11.0,
       39.5
      ],
      [
       11.5,
       50.0
      ],
      [
       12.0,
       33.5
      ],
      [
       12.5,
       22.5
      ],
      [
       13.0,
       22.4
      ],
      [
       13.5,
       20.1
      ],
      [
       14.0,
       19.3
      ],
      [
       14.5,
       20.3
      ],
      [
       15.0,
       24.0
      ],
      [
       15.5,
       29.6
      ]
     ]
    },
    {
     "id": 39,
     "cx": 0.17,
     "cy": 0.2851,
     "w": 0.0234,
     "h": 0.0605,
     "mean": 45.2,
     "low_pct": 88.9,
     "cue": "slouched, watching lesson",
     "n": 27,
     "series": [
      [
       1.5,
       39.8
      ],
      [
       2.0,
       40.7
      ],
      [
       2.5,
       41.0
      ],
      [
       3.0,
       45.8
      ],
      [
       3.5,
       48.9
      ],
      [
       4.0,
       51.7
      ]
     ]
    },
    {
     "id": 1,
     "cx": 0.4493,
     "cy": 0.5559,
     "w": 0.0716,
     "h": 0.2038,
     "mean": 52.2,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "n": 209,
     "series": [
      [
       1.5,
       52.7
      ],
      [
       2.0,
       53.7
      ],
      [
       2.5,
       51.5
      ],
      [
       3.0,
       50.5
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.5
      ],
      [
       6.0,
       53.4
      ],
      [
       6.5,
       56.5
      ],
      [
       7.0,
       54.8
      ],
      [
       7.5,
       52.8
      ],
      [
       8.0,
       52.8
      ],
      [
       8.5,
       50.7
      ],
      [
       9.0,
       52.9
      ],
      [
       9.5,
       52.6
      ],
      [
       10.0,
       54.9
      ],
      [
       10.5,
       54.0
      ],
      [
       11.0,
       53.4
      ],
      [
       11.5,
       53.5
      ],
      [
       12.0,
       52.1
      ],
      [
       12.5,
       50.9
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.4
      ],
      [
       14.0,
       52.1
      ],
      [
       14.5,
       52.1
      ],
      [
       15.0,
       53.3
      ],
      [
       15.5,
       51.7
      ],
      [
       16.0,
       50.2
      ],
      [
       16.5,
       50.0
      ],
      [
       17.0,
       50.0
      ],
      [
       17.5,
       50.7
      ],
      [
       18.0,
       53.7
      ],
      [
       18.5,
       54.2
      ],
      [
       19.0,
       54.9
      ],
      [
       19.5,
       55.6
      ],
      [
       20.0,
       54.5
      ],
      [
       20.5,
       50.4
      ],
      [
       21.0,
       50.5
      ],
      [
       21.5,
       50.8
      ],
      [
       22.0,
       54.0
      ],
      [
       22.5,
       54.3
      ]
     ]
    },
    {
     "id": 3,
     "cx": 0.297,
     "cy": 0.5952,
     "w": 0.0848,
     "h": 0.2303,
     "mean": 54.7,
     "low_pct": 0.0,
     "cue": "reading",
     "n": 209,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.8
      ],
      [
       5.5,
       52.9
      ],
      [
       6.0,
       53.7
      ],
      [
       6.5,
       55.4
      ],
      [
       7.0,
       56.5
      ],
      [
       7.5,
       58.4
      ],
      [
       8.0,
       58.0
      ],
      [
       8.5,
       53.8
      ],
      [
       9.0,
       54.4
      ],
      [
       9.5,
       55.7
      ],
      [
       10.0,
       61.2
      ],
      [
       10.5,
       62.2
      ],
      [
       11.0,
       62.3
      ],
      [
       11.5,
       58.5
      ],
      [
       12.0,
       56.0
      ],
      [
       12.5,
       54.2
      ],
      [
       13.0,
       52.9
      ],
      [
       13.5,
       52.3
      ],
      [
       14.0,
       52.7
      ],
      [
       14.5,
       51.5
      ],
      [
       15.0,
       54.3
      ],
      [
       15.5,
       55.5
      ],
      [
       16.0,
       59.6
      ],
      [
       16.5,
       60.8
      ],
      [
       17.0,
       59.7
      ],
      [
       17.5,
       57.4
      ],
      [
       18.0,
       56.8
      ],
      [
       18.5,
       57.2
      ],
      [
       19.0,
       57.8
      ],
      [
       19.5,
       56.8
      ],
      [
       20.0,
       55.3
      ],
      [
       20.5,
       51.5
      ],
      [
       21.0,
       50.1
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       50.0
      ],
      [
       22.5,
       50.2
      ]
     ]
    },
    {
     "id": 2,
     "cx": 0.7123,
     "cy": 0.4942,
     "w": 0.0582,
     "h": 0.1636,
     "mean": 40.4,
     "low_pct": 84.2,
     "cue": "reading / writing",
     "n": 209,
     "series": [
      [
       1.5,
       38.4
      ],
      [
       2.0,
       37.5
      ],
      [
       2.5,
       35.5
      ],
      [
       3.0,
       38.1
      ],
      [
       3.5,
       33.1
      ],
      [
       4.0,
       30.5
      ],
      [
       4.5,
       30.7
      ],
      [
       5.0,
       35.0
      ],
      [
       5.5,
       44.8
      ],
      [
       6.0,
       49.9
      ],
      [
       6.5,
       50.8
      ],
      [
       7.0,
       49.9
      ],
      [
       7.5,
       49.1
      ],
      [
       8.0,
       49.4
      ],
      [
       8.5,
       39.6
      ],
      [
       9.0,
       32.0
      ],
      [
       9.5,
       31.3
      ],
      [
       10.0,
       29.9
      ],
      [
       10.5,
       29.4
      ],
      [
       11.0,
       28.3
      ],
      [
       11.5,
       29.6
      ],
      [
       12.0,
       34.8
      ],
      [
       12.5,
       38.8
      ],
      [
       13.0,
       35.8
      ],
      [
       13.5,
       34.6
      ],
      [
       14.0,
       35.2
      ],
      [
       14.5,
       36.3
      ],
      [
       15.0,
       40.7
      ],
      [
       15.5,
       47.0
      ],
      [
       16.0,
       48.9
      ],
      [
       16.5,
       45.5
      ],
      [
       17.0,
       43.2
      ],
      [
       17.5,
       40.8
      ],
      [
       18.0,
       39.1
      ],
      [
       18.5,
       34.8
      ],
      [
       19.0,
       36.4
      ],
      [
       19.5,
       38.8
      ],
      [
       20.0,
       49.4
      ],
      [
       20.5,
       49.3
      ],
      [
       21.0,
       51.0
      ],
      [
       21.5,
       51.8
      ],
      [
       22.0,
       57.3
      ],
      [
       22.5,
       59.9
      ]
     ]
    },
    {
     "id": 5,
     "cx": 0.238,
     "cy": 0.403,
     "w": 0.0393,
     "h": 0.1294,
     "mean": 53.0,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "n": 209,
     "series": [
      [
       1.5,
       58.4
      ],
      [
       2.0,
       58.0
      ],
      [
       2.5,
       51.4
      ],
      [
       3.0,
       51.3
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.7
      ],
      [
       4.5,
       50.7
      ],
      [
       5.0,
       51.9
      ],
      [
       5.5,
       52.4
      ],
      [
       6.0,
       52.9
      ],
      [
       6.5,
       54.0
      ],
      [
       7.0,
       55.5
      ],
      [
       7.5,
       53.7
      ],
      [
       8.0,
       51.9
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.1
      ],
      [
       10.0,
       55.1
      ],
      [
       10.5,
       53.7
      ],
      [
       11.0,
       51.5
      ],
      [
       11.5,
       50.1
      ],
      [
       12.0,
       50.0
      ],
      [
       12.5,
       50.3
      ],
      [
       13.0,
       52.7
      ],
      [
       13.5,
       54.6
      ],
      [
       14.0,
       56.8
      ],
      [
       14.5,
       53.0
      ],
      [
       15.0,
       52.8
      ],
      [
       15.5,
       51.3
      ],
      [
       16.0,
       53.2
      ],
      [
       16.5,
       55.6
      ],
      [
       17.0,
       58.2
      ],
      [
       17.5,
       57.7
      ],
      [
       18.0,
       55.4
      ],
      [
       18.5,
       52.9
      ],
      [
       19.0,
       53.5
      ],
      [
       19.5,
       55.5
      ],
      [
       20.0,
       59.0
      ],
      [
       20.5,
       53.4
      ],
      [
       21.0,
       51.6
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       51.1
      ],
      [
       22.5,
       52.3
      ]
     ]
    },
    {
     "id": 6,
     "cx": 0.3443,
     "cy": 0.3983,
     "w": 0.0476,
     "h": 0.1109,
     "mean": 54.0,
     "low_pct": 0.0,
     "cue": "reading",
     "n": 209,
     "series": [
      [
       1.5,
       62.8
      ],
      [
       2.0,
       62.5
      ],
      [
       2.5,
       52.9
      ],
      [
       3.0,
       53.4
      ],
      [
       3.5,
       50.8
      ],
      [
       4.0,
       54.2
      ],
      [
       4.5,
       52.1
      ],
      [
       5.0,
       51.4
      ],
      [
       5.5,
       51.8
      ],
      [
       6.0,
       54.2
      ],
      [
       6.5,
       57.1
      ],
      [
       7.0,
       56.8
      ],
      [
       7.5,
       55.1
      ],
      [
       8.0,
       54.8
      ],
      [
       8.5,
       51.1
      ],
      [
       9.0,
       52.7
      ],
      [
       9.5,
       52.4
      ],
      [
       10.0,
       53.8
      ],
      [
       10.5,
       52.3
      ],
      [
       11.0,
       50.2
      ],
      [
       11.5,
       50.5
      ],
      [
       12.0,
       51.9
      ],
      [
       12.5,
       54.8
      ],
      [
       13.0,
       56.1
      ],
      [
       13.5,
       56.2
      ],
      [
       14.0,
       54.7
      ],
      [
       14.5,
       50.1
      ],
      [
       15.0,
       50.1
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       53.1
      ],
      [
       16.5,
       53.9
      ],
      [
       17.0,
       56.5
      ],
      [
       17.5,
       58.3
      ],
      [
       18.0,
       59.3
      ],
      [
       18.5,
       60.2
      ],
      [
       19.0,
       58.6
      ],
      [
       19.5,
       57.8
      ],
      [
       20.0,
       55.3
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.1
      ],
      [
       22.0,
       55.0
      ],
      [
       22.5,
       57.0
      ]
     ]
    },
    {
     "id": 4,
     "cx": 0.4988,
     "cy": 0.4446,
     "w": 0.0539,
     "h": 0.121,
     "mean": 51.7,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "n": 209,
     "series": [
      [
       1.5,
       51.6
      ],
      [
       2.0,
       54.6
      ],
      [
       2.5,
       51.9
      ],
      [
       3.0,
       53.2
      ],
      [
       3.5,
       50.5
      ],
      [
       4.0,
       52.3
      ],
      [
       4.5,
       50.2
      ],
      [
       5.0,
       50.1
      ],
      [
       5.5,
       50.4
      ],
      [
       6.0,
       51.8
      ],
      [
       6.5,
       53.1
      ],
      [
       7.0,
       53.2
      ],
      [
       7.5,
       50.9
      ],
      [
       8.0,
       51.0
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.1
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       50.2
      ],
      [
       10.5,
       50.0
      ],
      [
       11.0,
       51.2
      ],
      [
       11.5,
       51.6
      ],
      [
       12.0,
       51.4
      ],
      [
       12.5,
       50.7
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.0
      ],
      [
       14.0,
       50.3
      ],
      [
       14.5,
       50.6
      ],
      [
       15.0,
       51.9
      ],
      [
       15.5,
       50.4
      ],
      [
       16.0,
       52.7
      ],
      [
       16.5,
       51.4
      ],
      [
       17.0,
       52.3
      ],
      [
       17.5,
       52.1
      ],
      [
       18.0,
       52.4
      ],
      [
       18.5,
       53.3
      ],
      [
       19.0,
       54.1
      ],
      [
       19.5,
       55.7
      ],
      [
       20.0,
       55.6
      ],
      [
       20.5,
       50.5
      ],
      [
       21.0,
       50.4
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       53.6
      ],
      [
       22.5,
       55.2
      ]
     ]
    },
    {
     "id": 7,
     "cx": 0.3893,
     "cy": 0.4757,
     "w": 0.0547,
     "h": 0.1433,
     "mean": 53.4,
     "low_pct": 0.0,
     "cue": "reading",
     "n": 209,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.8
      ],
      [
       2.5,
       50.6
      ],
      [
       3.0,
       51.4
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.4
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.5
      ],
      [
       5.5,
       51.1
      ],
      [
       6.0,
       51.8
      ],
      [
       6.5,
       53.9
      ],
      [
       7.0,
       55.1
      ],
      [
       7.5,
       55.4
      ],
      [
       8.0,
       55.7
      ],
      [
       8.5,
       50.5
      ],
      [
       9.0,
       52.0
      ],
      [
       9.5,
       52.3
      ],
      [
       10.0,
       54.7
      ],
      [
       10.5,
       53.2
      ],
      [
       11.0,
       51.8
      ],
      [
       11.5,
       52.0
      ],
      [
       12.0,
       52.9
      ],
      [
       12.5,
       52.6
      ],
      [
       13.0,
       52.2
      ],
      [
       13.5,
       52.3
      ],
      [
       14.0,
       53.2
      ],
      [
       14.5,
       50.5
      ],
      [
       15.0,
       51.2
      ],
      [
       15.5,
       50.1
      ],
      [
       16.0,
       51.9
      ],
      [
       16.5,
       51.9
      ],
      [
       17.0,
       53.9
      ],
      [
       17.5,
       57.2
      ],
      [
       18.0,
       60.6
      ],
      [
       18.5,
       61.6
      ],
      [
       19.0,
       62.1
      ],
      [
       19.5,
       61.3
      ],
      [
       20.0,
       58.0
      ],
      [
       20.5,
       52.6
      ],
      [
       21.0,
       51.8
      ],
      [
       21.5,
       51.8
      ],
      [
       22.0,
       56.1
      ],
      [
       22.5,
       57.4
      ]
     ]
    },
    {
     "id": 11,
     "cx": 0.9869,
     "cy": 0.363,
     "w": 0.0261,
     "h": 0.0854,
     "mean": 50.1,
     "low_pct": 50.7,
     "cue": "reading / writing",
     "n": 209,
     "series": [
      [
       1.5,
       51.4
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       47.2
      ],
      [
       3.0,
       51.2
      ],
      [
       3.5,
       50.4
      ],
      [
       4.0,
       51.3
      ],
      [
       4.5,
       49.0
      ],
      [
       5.0,
       49.4
      ],
      [
       5.5,
       51.1
      ],
      [
       6.0,
       54.9
      ],
      [
       6.5,
       55.5
      ],
      [
       7.0,
       56.1
      ],
      [
       7.5,
       55.4
      ],
      [
       8.0,
       57.5
      ],
      [
       8.5,
       52.8
      ],
      [
       9.0,
       52.3
      ],
      [
       9.5,
       51.4
      ],
      [
       10.0,
       52.9
      ],
      [
       10.5,
       53.5
      ],
      [
       11.0,
       55.0
      ],
      [
       11.5,
       56.6
      ],
      [
       12.0,
       59.6
      ],
      [
       12.5,
       60.2
      ],
      [
       13.0,
       55.9
      ],
      [
       13.5,
       49.4
      ],
      [
       14.0,
       46.2
      ],
      [
       14.5,
       43.9
      ],
      [
       15.0,
       46.5
      ],
      [
       15.5,
       46.1
      ],
      [
       16.0,
       48.4
      ],
      [
       16.5,
       49.0
      ],
      [
       17.0,
       51.8
      ],
      [
       17.5,
       51.8
      ],
      [
       18.0,
       44.0
      ],
      [
       18.5,
       40.6
      ],
      [
       19.0,
       39.7
      ],
      [
       19.5,
       43.6
      ],
      [
       20.0,
       45.8
      ],
      [
       20.5,
       43.5
      ],
      [
       21.0,
       43.4
      ],
      [
       21.5,
       44.0
      ],
      [
       22.0,
       48.0
      ],
      [
       22.5,
       49.5
      ]
     ]
    },
    {
     "id": 8,
     "cx": 0.303,
     "cy": 0.3512,
     "w": 0.0438,
     "h": 0.0751,
     "mean": 53.9,
     "low_pct": 0.0,
     "cue": "reading",
     "n": 209,
     "series": [
      [
       1.5,
       57.8
      ],
      [
       2.0,
       56.5
      ],
      [
       2.5,
       50.6
      ],
      [
       3.0,
       52.1
      ],
      [
       3.5,
       51.9
      ],
      [
       4.0,
       53.8
      ],
      [
       4.5,
       50.8
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.6
      ],
      [
       6.0,
       53.0
      ],
      [
       6.5,
       56.4
      ],
      [
       7.0,
       58.1
      ],
      [
       7.5,
       57.3
      ],
      [
       8.0,
       57.4
      ],
      [
       8.5,
       51.8
      ],
      [
       9.0,
       52.2
      ],
      [
       9.5,
       51.9
      ],
      [
       10.0,
       56.3
      ],
      [
       10.5,
       57.5
      ],
      [
       11.0,
       57.1
      ],
      [
       11.5,
       56.1
      ],
      [
       12.0,
       55.6
      ],
      [
       12.5,
       53.9
      ],
      [
       13.0,
       53.2
      ],
      [
       13.5,
       53.3
      ],
      [
       14.0,
       52.9
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       53.2
      ],
      [
       16.5,
       54.8
      ],
      [
       17.0,
       54.4
      ],
      [
       17.5,
       56.0
      ],
      [
       18.0,
       56.5
      ],
      [
       18.5,
       60.6
      ],
      [
       19.0,
       60.2
      ],
      [
       19.5,
       58.9
      ],
      [
       20.0,
       55.5
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       51.2
      ],
      [
       22.5,
       53.2
      ]
     ]
    },
    {
     "id": 9,
     "cx": 0.4305,
     "cy": 0.386,
     "w": 0.0467,
     "h": 0.1054,
     "mean": 50.0,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "n": 209,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.2
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.3
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.0
      ],
      [
       7.0,
       50.0
      ],
      [
       7.5,
       50.0
      ],
      [
       8.0,
       50.0
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       50.0
      ],
      [
       10.5,
       50.0
      ],
      [
       11.0,
       50.0
      ],
      [
       11.5,
       50.0
      ],
      [
       12.0,
       50.0
      ],
      [
       12.5,
       50.0
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.0
      ],
      [
       14.0,
       50.0
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       50.0
      ],
      [
       16.5,
       50.0
      ],
      [
       17.0,
       50.0
      ],
      [
       17.5,
       50.0
      ],
      [
       18.0,
       50.0
      ],
      [
       18.5,
       50.0
      ],
      [
       19.0,
       50.0
      ],
      [
       19.5,
       50.0
      ],
      [
       20.0,
       50.6
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       50.2
      ],
      [
       22.5,
       50.0
      ]
     ]
    },
    {
     "id": 10,
     "cx": 0.4616,
     "cy": 0.3343,
     "w": 0.0395,
     "h": 0.0889,
     "mean": 51.1,
     "low_pct": 0.0,
     "cue": "reading",
     "n": 209,
     "series": [
      [
       1.5,
       54.7
      ],
      [
       2.0,
       54.7
      ],
      [
       2.5,
       50.4
      ],
      [
       3.0,
       51.6
      ],
      [
       3.5,
       50.5
      ],
      [
       4.0,
       52.7
      ],
      [
       4.5,
       50.2
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       50.4
      ],
      [
       6.5,
       52.9
      ],
      [
       7.0,
       54.9
      ],
      [
       7.5,
       55.8
      ],
      [
       8.0,
       55.4
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       50.0
      ],
      [
       10.5,
       50.0
      ],
      [
       11.0,
       50.0
      ],
      [
       11.5,
       50.0
      ],
      [
       12.0,
       50.0
      ],
      [
       12.5,
       50.0
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.0
      ],
      [
       14.0,
       50.0
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       50.1
      ],
      [
       16.5,
       50.0
      ],
      [
       17.0,
       50.5
      ],
      [
       17.5,
       51.3
      ],
      [
       18.0,
       50.4
      ],
      [
       18.5,
       50.0
      ],
      [
       19.0,
       50.1
      ],
      [
       19.5,
       51.4
      ],
      [
       20.0,
       52.6
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       54.2
      ],
      [
       22.5,
       55.7
      ]
     ]
    },
    {
     "id": 15,
     "cx": 0.5782,
     "cy": 0.438,
     "w": 0.0321,
     "h": 0.1148,
     "mean": 51.8,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "n": 209,
     "series": [
      [
       1.5,
       54.1
      ],
      [
       2.0,
       53.3
      ],
      [
       2.5,
       50.2
      ],
      [
       3.0,
       50.9
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       51.0
      ],
      [
       4.5,
       50.1
      ],
      [
       5.0,
       51.1
      ],
      [
       5.5,
       50.7
      ],
      [
       6.0,
       50.2
      ],
      [
       6.5,
       51.6
      ],
      [
       7.0,
       54.1
      ],
      [
       7.5,
       58.5
      ],
      [
       8.0,
       61.5
      ],
      [
       8.5,
       54.7
      ],
      [
       9.0,
       53.9
      ],
      [
       9.5,
       52.3
      ],
      [
       10.0,
       55.5
      ],
      [
       10.5,
       54.0
      ],
      [
       11.0,
       52.1
      ],
      [
       11.5,
       50.2
      ],
      [
       12.0,
       50.3
      ],
      [
       12.5,
       50.0
      ],
      [
       13.0,
       51.0
      ],
      [
       13.5,
       50.0
      ],
      [
       14.0,
       50.9
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       50.3
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       50.0
      ],
      [
       16.5,
       50.0
      ],
      [
       17.0,
       50.0
      ],
      [
       17.5,
       51.0
      ],
      [
       18.0,
       52.8
      ],
      [
       18.5,
       53.5
      ],
      [
       19.0,
       53.8
      ],
      [
       19.5,
       52.2
      ],
      [
       20.0,
       50.4
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       52.8
      ],
      [
       22.5,
       51.4
      ]
     ]
    },
    {
     "id": 14,
     "cx": 0.5963,
     "cy": 0.5387,
     "w": 0.0745,
     "h": 0.1569,
     "mean": 53.5,
     "low_pct": 9.1,
     "cue": "reading / taking notes",
     "n": 209,
     "series": [
      [
       1.5,
       51.6
      ],
      [
       2.0,
       51.6
      ],
      [
       2.5,
       49.4
      ],
      [
       3.0,
       51.7
      ],
      [
       3.5,
       51.4
      ],
      [
       4.0,
       53.4
      ],
      [
       4.5,
       51.7
      ],
      [
       5.0,
       51.4
      ],
      [
       5.5,
       53.3
      ],
      [
       6.0,
       54.7
      ],
      [
       6.5,
       57.8
      ],
      [
       7.0,
       57.9
      ],
      [
       7.5,
       59.0
      ],
      [
       8.0,
       59.1
      ],
      [
       8.5,
       53.8
      ],
      [
       9.0,
       54.3
      ],
      [
       9.5,
       54.4
      ],
      [
       10.0,
       58.1
      ],
      [
       10.5,
       57.4
      ],
      [
       11.0,
       56.2
      ],
      [
       11.5,
       54.0
      ],
      [
       12.0,
       51.3
      ],
      [
       12.5,
       49.9
      ],
      [
       13.0,
       49.9
      ],
      [
       13.5,
       51.1
      ],
      [
       14.0,
       53.3
      ],
      [
       14.5,
       51.0
      ],
      [
       15.0,
       52.5
      ],
      [
       15.5,
       51.5
      ],
      [
       16.0,
       53.0
      ],
      [
       16.5,
       53.1
      ],
      [
       17.0,
       54.7
      ],
      [
       17.5,
       57.8
      ],
      [
       18.0,
       61.3
      ],
      [
       18.5,
       62.7
      ],
      [
       19.0,
       54.1
      ],
      [
       19.5,
       51.3
      ],
      [
       20.0,
       50.0
      ],
      [
       20.5,
       47.0
      ],
      [
       21.0,
       47.9
      ],
      [
       21.5,
       49.6
      ],
      [
       22.0,
       51.0
      ],
      [
       22.5,
       51.2
      ]
     ]
    },
    {
     "id": 25,
     "cx": 0.2777,
     "cy": 0.3031,
     "w": 0.0341,
     "h": 0.0856,
     "mean": 50.8,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "n": 209,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       52.9
      ],
      [
       4.5,
       51.6
      ],
      [
       5.0,
       51.0
      ],
      [
       5.5,
       51.3
      ],
      [
       6.0,
       51.9
      ],
      [
       6.5,
       50.4
      ],
      [
       7.0,
       50.0
      ],
      [
       7.5,
       50.0
      ],
      [
       8.0,
       50.0
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       50.0
      ],
      [
       10.5,
       50.0
      ],
      [
       11.0,
       50.0
      ],
      [
       11.5,
       50.6
      ],
      [
       12.0,
       54.3
      ],
      [
       12.5,
       55.6
      ],
      [
       13.0,
       57.0
      ],
      [
       13.5,
       53.6
      ],
      [
       14.0,
       50.1
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       50.0
      ],
      [
       16.5,
       50.0
      ],
      [
       17.0,
       50.1
      ],
      [
       17.5,
       50.7
      ],
      [
       18.0,
       50.0
      ],
      [
       18.5,
       50.0
      ],
      [
       19.0,
       50.0
      ],
      [
       19.5,
       50.0
      ],
      [
       20.0,
       51.1
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       51.8
      ],
      [
       22.5,
       52.1
      ]
     ]
    },
    {
     "id": 23,
     "cx": 0.7413,
     "cy": 0.3025,
     "w": 0.0216,
     "h": 0.0622,
     "mean": 34.5,
     "low_pct": 88.9,
     "cue": "reading / writing",
     "n": 45,
     "series": [
      [
       1.5,
       49.3
      ],
      [
       2.0,
       51.6
      ],
      [
       2.5,
       46.1
      ],
      [
       3.0,
       40.6
      ],
      [
       3.5,
       30.8
      ],
      [
       4.0,
       24.5
      ],
      [
       4.5,
       22.9
      ],
      [
       5.0,
       23.6
      ],
      [
       5.5,
       25.0
      ],
      [
       6.0,
       29.3
      ],
      [
       6.5,
       34.2
      ],
      [
       7.0,
       35.1
      ]
     ]
    },
    {
     "id": 24,
     "cx": 0.184,
     "cy": 0.3061,
     "w": 0.0334,
     "h": 0.0977,
     "mean": 50.0,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "n": 209,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.0
      ],
      [
       7.0,
       50.0
      ],
      [
       7.5,
       50.0
      ],
      [
       8.0,
       50.0
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       50.0
      ],
      [
       10.5,
       50.0
      ],
      [
       11.0,
       50.0
      ],
      [
       11.5,
       50.0
      ],
      [
       12.0,
       50.0
      ],
      [
       12.5,
       50.0
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.0
      ],
      [
       14.0,
       50.0
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       50.0
      ],
      [
       16.5,
       50.0
      ],
      [
       17.0,
       50.0
      ],
      [
       17.5,
       50.0
      ],
      [
       18.0,
       50.0
      ],
      [
       18.5,
       50.0
      ],
      [
       19.0,
       50.0
      ],
      [
       19.5,
       50.0
      ],
      [
       20.0,
       50.6
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       50.2
      ],
      [
       22.5,
       51.1
      ]
     ]
    },
    {
     "id": 21,
     "cx": 0.4066,
     "cy": 0.2916,
     "w": 0.0332,
     "h": 0.0816,
     "mean": 49.9,
     "low_pct": 5.7,
     "cue": "reading",
     "n": 209,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.1
      ],
      [
       7.0,
       51.4
      ],
      [
       7.5,
       51.3
      ],
      [
       8.0,
       51.2
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       50.0
      ],
      [
       10.5,
       50.0
      ],
      [
       11.0,
       50.0
      ],
      [
       11.5,
       50.0
      ],
      [
       12.0,
       50.0
      ],
      [
       12.5,
       50.0
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.0
      ],
      [
       14.0,
       50.0
      ],
      [
       14.5,
       48.4
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       44.1
      ],
      [
       16.0,
       50.0
      ],
      [
       16.5,
       49.9
      ],
      [
       17.0,
       48.3
      ],
      [
       17.5,
       50.0
      ],
      [
       18.0,
       50.0
      ],
      [
       18.5,
       50.0
      ],
      [
       19.0,
       50.0
      ],
      [
       19.5,
       50.0
      ],
      [
       20.0,
       50.0
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       50.0
      ],
      [
       22.5,
       50.0
      ]
     ]
    },
    {
     "id": 12,
     "cx": 0.2691,
     "cy": 0.4731,
     "w": 0.0712,
     "h": 0.2055,
     "mean": 53.6,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "n": 209,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.7
      ],
      [
       5.5,
       51.8
      ],
      [
       6.0,
       52.4
      ],
      [
       6.5,
       54.1
      ],
      [
       7.0,
       55.6
      ],
      [
       7.5,
       55.8
      ],
      [
       8.0,
       55.7
      ],
      [
       8.5,
       51.5
      ],
      [
       9.0,
       52.4
      ],
      [
       9.5,
       53.6
      ],
      [
       10.0,
       57.5
      ],
      [
       10.5,
       58.2
      ],
      [
       11.0,
       59.2
      ],
      [
       11.5,
       57.3
      ],
      [
       12.0,
       54.5
      ],
      [
       12.5,
       52.5
      ],
      [
       13.0,
       50.8
      ],
      [
       13.5,
       51.1
      ],
      [
       14.0,
       52.4
      ],
      [
       14.5,
       51.5
      ],
      [
       15.0,
       53.7
      ],
      [
       15.5,
       53.4
      ],
      [
       16.0,
       56.5
      ],
      [
       16.5,
       58.3
      ],
      [
       17.0,
       58.8
      ],
      [
       17.5,
       56.8
      ],
      [
       18.0,
       56.0
      ],
      [
       18.5,
       56.1
      ],
      [
       19.0,
       57.7
      ],
      [
       19.5,
       57.3
      ],
      [
       20.0,
       56.3
      ],
      [
       20.5,
       51.2
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       50.0
      ],
      [
       22.5,
       50.5
      ]
     ]
    },
    {
     "id": 13,
     "cx": 0.2107,
     "cy": 0.351,
     "w": 0.0368,
     "h": 0.1102,
     "mean": 50.1,
     "low_pct": 0.0,
     "cue": "reading",
     "n": 209,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.8
      ],
      [
       7.0,
       50.3
      ],
      [
       7.5,
       50.0
      ],
      [
       8.0,
       50.0
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       51.3
      ],
      [
       10.5,
       50.0
      ],
      [
       11.0,
       50.0
      ],
      [
       11.5,
       50.0
      ],
      [
       12.0,
       50.0
      ],
      [
       12.5,
       50.0
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.0
      ],
      [
       14.0,
       50.0
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       50.0
      ],
      [
       16.5,
       50.0
      ],
      [
       17.0,
       50.0
      ],
      [
       17.5,
       50.0
      ],
      [
       18.0,
       50.0
      ],
      [
       18.5,
       50.0
      ],
      [
       19.0,
       50.0
      ],
      [
       19.5,
       50.0
      ],
      [
       20.0,
       50.0
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       50.0
      ],
      [
       22.5,
       50.0
      ]
     ]
    },
    {
     "id": 18,
     "cx": 0.2533,
     "cy": 0.2672,
     "w": 0.0245,
     "h": 0.0881,
     "mean": 50.2,
     "low_pct": 45.5,
     "cue": "reading / taking notes",
     "n": 209,
     "series": [
      [
       1.5,
       51.1
      ],
      [
       2.0,
       53.4
      ],
      [
       2.5,
       48.0
      ],
      [
       3.0,
       50.5
      ],
      [
       3.5,
       50.5
      ],
      [
       4.0,
       53.0
      ],
      [
       4.5,
       52.5
      ],
      [
       5.0,
       50.3
      ],
      [
       5.5,
       49.6
      ],
      [
       6.0,
       48.8
      ],
      [
       6.5,
       47.8
      ],
      [
       7.0,
       47.4
      ],
      [
       7.5,
       45.6
      ],
      [
       8.0,
       46.2
      ],
      [
       8.5,
       44.3
      ],
      [
       9.0,
       44.3
      ],
      [
       9.5,
       43.4
      ],
      [
       10.0,
       44.4
      ],
      [
       10.5,
       46.2
      ],
      [
       11.0,
       49.1
      ],
      [
       11.5,
       52.5
      ],
      [
       12.0,
       54.8
      ],
      [
       12.5,
       53.3
      ],
      [
       13.0,
       53.3
      ],
      [
       13.5,
       52.4
      ],
      [
       14.0,
       52.7
      ],
      [
       14.5,
       46.0
      ],
      [
       15.0,
       43.9
      ],
      [
       15.5,
       43.2
      ],
      [
       16.0,
       45.8
      ],
      [
       16.5,
       49.0
      ],
      [
       17.0,
       51.8
      ],
      [
       17.5,
       54.2
      ],
      [
       18.0,
       52.5
      ],
      [
       18.5,
       51.1
      ],
      [
       19.0,
       52.3
      ],
      [
       19.5,
       57.7
      ],
      [
       20.0,
       63.7
      ],
      [
       20.5,
       54.4
      ],
      [
       21.0,
       52.2
      ],
      [
       21.5,
       49.1
      ],
      [
       22.0,
       54.0
      ],
      [
       22.5,
       57.0
      ]
     ]
    },
    {
     "id": 20,
     "cx": 0.3172,
     "cy": 0.2655,
     "w": 0.027,
     "h": 0.0762,
     "mean": 52.0,
     "low_pct": 0.5,
     "cue": "reading / taking notes",
     "n": 209,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       51.4
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.2
      ],
      [
       3.5,
       50.1
      ],
      [
       4.0,
       53.5
      ],
      [
       4.5,
       52.4
      ],
      [
       5.0,
       51.5
      ],
      [
       5.5,
       50.7
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.0
      ],
      [
       7.0,
       50.1
      ],
      [
       7.5,
       52.8
      ],
      [
       8.0,
       54.8
      ],
      [
       8.5,
       50.5
      ],
      [
       9.0,
       51.3
      ],
      [
       9.5,
       51.1
      ],
      [
       10.0,
       54.3
      ],
      [
       10.5,
       52.4
      ],
      [
       11.0,
       52.5
      ],
      [
       11.5,
       51.5
      ],
      [
       12.0,
       51.8
      ],
      [
       12.5,
       50.1
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.5
      ],
      [
       14.0,
       54.1
      ],
      [
       14.5,
       50.7
      ],
      [
       15.0,
       50.9
      ],
      [
       15.5,
       50.1
      ],
      [
       16.0,
       52.3
      ],
      [
       16.5,
       51.8
      ],
      [
       17.0,
       51.1
      ],
      [
       17.5,
       52.1
      ],
      [
       18.0,
       54.1
      ],
      [
       18.5,
       57.2
      ],
      [
       19.0,
       58.1
      ],
      [
       19.5,
       59.4
      ],
      [
       20.0,
       58.8
      ],
      [
       20.5,
       52.5
      ],
      [
       21.0,
       50.8
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       50.0
      ],
      [
       22.5,
       41.9
      ]
     ]
    },
    {
     "id": 26,
     "cx": 0.9869,
     "cy": 0.4814,
     "w": 0.0262,
     "h": 0.0948,
     "mean": 49.1,
     "low_pct": 7.7,
     "cue": "reading / writing",
     "n": 209,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       46.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       50.0
      ],
      [
       7.0,
       50.0
      ],
      [
       7.5,
       50.0
      ],
      [
       8.0,
       50.0
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.3
      ],
      [
       9.5,
       52.8
      ],
      [
       10.0,
       57.9
      ],
      [
       10.5,
       56.6
      ],
      [
       11.0,
       53.5
      ],
      [
       11.5,
       49.7
      ],
      [
       12.0,
       50.0
      ],
      [
       12.5,
       50.0
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.0
      ],
      [
       14.0,
       50.0
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       50.0
      ],
      [
       16.5,
       50.0
      ],
      [
       17.0,
       50.0
      ],
      [
       17.5,
       46.0
      ],
      [
       18.0,
       50.0
      ],
      [
       18.5,
       43.6
      ],
      [
       19.0,
       34.0
      ],
      [
       19.5,
       30.4
      ],
      [
       20.0,
       38.8
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       50.1
      ],
      [
       22.5,
       51.9
      ]
     ]
    },
    {
     "id": 22,
     "cx": 0.5153,
     "cy": 0.378,
     "w": 0.0433,
     "h": 0.1014,
     "mean": 53.0,
     "low_pct": 0.0,
     "cue": "reading / writing",
     "n": 209,
     "series": [
      [
       1.5,
       55.4
      ],
      [
       2.0,
       58.1
      ],
      [
       2.5,
       52.2
      ],
      [
       3.0,
       52.7
      ],
      [
       3.5,
       50.7
      ],
      [
       4.0,
       53.7
      ],
      [
       4.5,
       52.4
      ],
      [
       5.0,
       52.3
      ],
      [
       5.5,
       51.9
      ],
      [
       6.0,
       52.6
      ],
      [
       6.5,
       53.8
      ],
      [
       7.0,
       56.0
      ],
      [
       7.5,
       54.8
      ],
      [
       8.0,
       54.3
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.3
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       51.1
      ],
      [
       10.5,
       50.1
      ],
      [
       11.0,
       52.7
      ],
      [
       11.5,
       53.9
      ],
      [
       12.0,
       54.5
      ],
      [
       12.5,
       53.5
      ],
      [
       13.0,
       51.1
      ],
      [
       13.5,
       51.5
      ],
      [
       14.0,
       53.6
      ],
      [
       14.5,
       51.2
      ],
      [
       15.0,
       51.6
      ],
      [
       15.5,
       50.1
      ],
      [
       16.0,
       51.7
      ],
      [
       16.5,
       50.6
      ],
      [
       17.0,
       51.4
      ],
      [
       17.5,
       52.4
      ],
      [
       18.0,
       54.5
      ],
      [
       18.5,
       57.7
      ],
      [
       19.0,
       61.6
      ],
      [
       19.5,
       60.8
      ],
      [
       20.0,
       57.6
      ],
      [
       20.5,
       50.4
      ],
      [
       21.0,
       50.2
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       52.9
      ],
      [
       22.5,
       54.6
      ]
     ]
    },
    {
     "id": 32,
     "cx": 0.9341,
     "cy": 0.3878,
     "w": 0.0285,
     "h": 0.0646,
     "mean": 43.6,
     "low_pct": 72.2,
     "cue": "slouched, watching lesson",
     "n": 209,
     "series": [
      [
       1.5,
       50.1
      ],
      [
       2.0,
       52.0
      ],
      [
       2.5,
       47.0
      ],
      [
       3.0,
       45.2
      ],
      [
       3.5,
       40.7
      ],
      [
       4.0,
       37.6
      ],
      [
       4.5,
       36.7
      ],
      [
       5.0,
       35.9
      ],
      [
       5.5,
       34.1
      ],
      [
       6.0,
       32.0
      ],
      [
       6.5,
       31.0
      ],
      [
       7.0,
       33.9
      ],
      [
       7.5,
       36.4
      ],
      [
       8.0,
       40.1
      ],
      [
       8.5,
       40.8
      ],
      [
       9.0,
       42.6
      ],
      [
       9.5,
       45.0
      ],
      [
       10.0,
       46.9
      ],
      [
       10.5,
       49.1
      ],
      [
       11.0,
       50.7
      ],
      [
       11.5,
       53.0
      ],
      [
       12.0,
       52.8
      ],
      [
       12.5,
       49.3
      ],
      [
       13.0,
       48.0
      ],
      [
       13.5,
       49.4
      ],
      [
       14.0,
       52.2
      ],
      [
       14.5,
       46.4
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       44.3
      ],
      [
       16.0,
       49.1
      ],
      [
       16.5,
       50.0
      ],
      [
       17.0,
       49.2
      ],
      [
       17.5,
       49.4
      ],
      [
       18.0,
       51.0
      ],
      [
       18.5,
       51.8
      ],
      [
       19.0,
       49.3
      ],
      [
       19.5,
       50.0
      ],
      [
       20.0,
       40.4
      ],
      [
       20.5,
       32.2
      ],
      [
       21.0,
       31.6
      ],
      [
       21.5,
       31.2
      ],
      [
       22.0,
       34.6
      ],
      [
       22.5,
       35.7
      ]
     ]
    },
    {
     "id": 35,
     "cx": 0.3425,
     "cy": 0.3032,
     "w": 0.0346,
     "h": 0.084,
     "mean": 49.3,
     "low_pct": 56.9,
     "cue": "reading / taking notes",
     "n": 209,
     "series": [
      [
       1.5,
       40.7
      ],
      [
       2.0,
       43.6
      ],
      [
       2.5,
       44.5
      ],
      [
       3.0,
       46.2
      ],
      [
       3.5,
       48.7
      ],
      [
       4.0,
       52.5
      ],
      [
       4.5,
       50.7
      ],
      [
       5.0,
       50.3
      ],
      [
       5.5,
       50.1
      ],
      [
       6.0,
       50.4
      ],
      [
       6.5,
       50.5
      ],
      [
       7.0,
       50.2
      ],
      [
       7.5,
       49.4
      ],
      [
       8.0,
       51.1
      ],
      [
       8.5,
       49.1
      ],
      [
       9.0,
       50.5
      ],
      [
       9.5,
       50.5
      ],
      [
       10.0,
       52.5
      ],
      [
       10.5,
       52.5
      ],
      [
       11.0,
       51.9
      ],
      [
       11.5,
       51.1
      ],
      [
       12.0,
       49.6
      ],
      [
       12.5,
       47.0
      ],
      [
       13.0,
       45.2
      ],
      [
       13.5,
       45.0
      ],
      [
       14.0,
       47.4
      ],
      [
       14.5,
       48.1
      ],
      [
       15.0,
       50.7
      ],
      [
       15.5,
       50.8
      ],
      [
       16.0,
       53.2
      ],
      [
       16.5,
       51.4
      ],
      [
       17.0,
       48.2
      ],
      [
       17.5,
       47.2
      ],
      [
       18.0,
       48.8
      ],
      [
       18.5,
       51.7
      ],
      [
       19.0,
       50.8
      ],
      [
       19.5,
       49.4
      ],
      [
       20.0,
       48.8
      ],
      [
       20.5,
       47.7
      ],
      [
       21.0,
       49.3
      ],
      [
       21.5,
       48.4
      ],
      [
       22.0,
       48.4
      ],
      [
       22.5,
       47.6
      ]
     ]
    },
    {
     "id": 33,
     "cx": 0.9386,
     "cy": 0.5055,
     "w": 0.0249,
     "h": 0.0674,
     "mean": 35.6,
     "low_pct": 48.8,
     "cue": "reading",
     "n": 209,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       38.0
      ],
      [
       3.0,
       22.8
      ],
      [
       3.5,
       23.6
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       44.0
      ],
      [
       5.0,
       38.0
      ],
      [
       5.5,
       32.0
      ],
      [
       6.0,
       50.0
      ],
      [
       6.5,
       44.0
      ],
      [
       7.0,
       50.0
      ],
      [
       7.5,
       50.0
      ],
      [
       8.0,
       50.0
      ],
      [
       8.5,
       20.0
      ],
      [
       9.0,
       32.0
      ],
      [
       9.5,
       31.2
      ],
      [
       10.0,
       38.0
      ],
      [
       10.5,
       44.0
      ],
      [
       11.0,
       37.2
      ],
      [
       11.5,
       26.0
      ],
      [
       12.0,
       33.1
      ],
      [
       12.5,
       40.0
      ],
      [
       13.0,
       27.2
      ],
      [
       13.5,
       16.8
      ],
      [
       14.0,
       20.0
      ],
      [
       14.5,
       26.0
      ],
      [
       15.0,
       32.0
      ],
      [
       15.5,
       32.0
      ],
      [
       16.0,
       32.9
      ],
      [
       16.5,
       32.5
      ],
      [
       17.0,
       45.7
      ],
      [
       17.5,
       47.2
      ],
      [
       18.0,
       48.2
      ],
      [
       18.5,
       42.2
      ],
      [
       19.0,
       26.8
      ],
      [
       19.5,
       20.0
      ],
      [
       20.0,
       32.1
      ],
      [
       20.5,
       26.0
      ],
      [
       21.0,
       32.0
      ],
      [
       21.5,
       32.0
      ],
      [
       22.0,
       32.3
      ],
      [
       22.5,
       50.1
      ]
     ]
    },
    {
     "id": 19,
     "cx": 0.6969,
     "cy": 0.3279,
     "w": 0.0272,
     "h": 0.0676,
     "mean": 27.4,
     "low_pct": 80.2,
     "cue": "glancing right",
     "n": 116,
     "series": [
      [
       1.5,
       50.0
      ],
      [
       2.0,
       50.0
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       43.9
      ],
      [
       3.5,
       18.8
      ],
      [
       4.0,
       17.6
      ],
      [
       4.5,
       17.7
      ],
      [
       5.0,
       19.4
      ],
      [
       5.5,
       26.0
      ],
      [
       6.0,
       28.4
      ],
      [
       6.5,
       31.3
      ],
      [
       7.0,
       33.1
      ],
      [
       7.5,
       32.0
      ],
      [
       8.0,
       32.0
      ],
      [
       8.5,
       25.1
      ],
      [
       9.0,
       31.9
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       23.7
      ],
      [
       10.5,
       14.4
      ],
      [
       11.0,
       11.6
      ],
      [
       11.5,
       11.7
      ],
      [
       12.0,
       14.2
      ],
      [
       12.5,
       19.8
      ],
      [
       13.0,
       23.4
      ],
      [
       13.5,
       22.1
      ]
     ]
    },
    {
     "id": 29,
     "cx": 0.383,
     "cy": 0.3319,
     "w": 0.0404,
     "h": 0.0962,
     "mean": 52.5,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "n": 209,
     "series": [
      [
       1.5,
       59.1
      ],
      [
       2.0,
       59.2
      ],
      [
       2.5,
       51.9
      ],
      [
       3.0,
       52.8
      ],
      [
       3.5,
       50.3
      ],
      [
       4.0,
       51.9
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.1
      ],
      [
       6.0,
       50.3
      ],
      [
       6.5,
       51.6
      ],
      [
       7.0,
       52.6
      ],
      [
       7.5,
       53.7
      ],
      [
       8.0,
       53.5
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.5
      ],
      [
       9.5,
       51.0
      ],
      [
       10.0,
       54.9
      ],
      [
       10.5,
       54.4
      ],
      [
       11.0,
       54.4
      ],
      [
       11.5,
       55.1
      ],
      [
       12.0,
       56.8
      ],
      [
       12.5,
       56.7
      ],
      [
       13.0,
       58.0
      ],
      [
       13.5,
       54.5
      ],
      [
       14.0,
       52.2
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       51.0
      ],
      [
       16.5,
       51.5
      ],
      [
       17.0,
       53.1
      ],
      [
       17.5,
       53.9
      ],
      [
       18.0,
       53.4
      ],
      [
       18.5,
       52.6
      ],
      [
       19.0,
       52.5
      ],
      [
       19.5,
       54.5
      ],
      [
       20.0,
       53.5
      ],
      [
       20.5,
       50.1
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       50.0
      ],
      [
       22.5,
       50.0
      ]
     ]
    },
    {
     "id": 31,
     "cx": 0.6662,
     "cy": 0.2391,
     "w": 0.0222,
     "h": 0.0614,
     "mean": 51.4,
     "low_pct": 0.0,
     "cue": "reading",
     "n": 193,
     "series": [
      [
       1.5,
       58.4
      ],
      [
       2.0,
       61.1
      ],
      [
       2.5,
       51.5
      ],
      [
       3.0,
       50.9
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       51.2
      ],
      [
       4.5,
       50.6
      ],
      [
       5.0,
       52.2
      ],
      [
       5.5,
       52.5
      ],
      [
       6.0,
       53.5
      ],
      [
       6.5,
       56.3
      ],
      [
       7.0,
       55.0
      ],
      [
       7.5,
       52.2
      ],
      [
       8.0,
       51.3
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       52.0
      ],
      [
       10.5,
       55.5
      ],
      [
       11.0,
       54.6
      ],
      [
       11.5,
       51.2
      ],
      [
       12.0,
       50.0
      ],
      [
       12.5,
       50.0
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.0
      ],
      [
       14.0,
       50.0
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       50.0
      ],
      [
       16.5,
       50.0
      ],
      [
       17.0,
       50.0
      ],
      [
       17.5,
       50.0
      ],
      [
       18.0,
       50.0
      ],
      [
       18.5,
       50.0
      ],
      [
       19.0,
       50.0
      ],
      [
       19.5,
       50.0
      ],
      [
       20.0,
       50.0
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ]
     ]
    },
    {
     "id": 27,
     "cx": 0.8305,
     "cy": 0.3463,
     "w": 0.0295,
     "h": 0.0884,
     "mean": 50.6,
     "low_pct": 1.5,
     "cue": "reading",
     "n": 206,
     "series": [
      [
       2.0,
       16.5
      ],
      [
       2.5,
       50.0
      ],
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       50.0
      ],
      [
       4.5,
       50.0
      ],
      [
       5.0,
       50.0
      ],
      [
       5.5,
       50.0
      ],
      [
       6.0,
       50.4
      ],
      [
       6.5,
       52.9
      ],
      [
       7.0,
       52.8
      ],
      [
       7.5,
       51.7
      ],
      [
       8.0,
       50.5
      ],
      [
       8.5,
       50.0
      ],
      [
       9.0,
       50.0
      ],
      [
       9.5,
       50.0
      ],
      [
       10.0,
       50.0
      ],
      [
       10.5,
       50.0
      ],
      [
       11.0,
       50.0
      ],
      [
       11.5,
       50.0
      ],
      [
       12.0,
       50.0
      ],
      [
       12.5,
       50.0
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.7
      ],
      [
       14.0,
       53.8
      ],
      [
       14.5,
       50.2
      ],
      [
       15.0,
       50.6
      ],
      [
       15.5,
       50.1
      ],
      [
       16.0,
       52.3
      ],
      [
       16.5,
       51.1
      ],
      [
       17.0,
       51.0
      ],
      [
       17.5,
       52.2
      ],
      [
       18.0,
       54.9
      ],
      [
       18.5,
       57.1
      ],
      [
       19.0,
       55.5
      ],
      [
       19.5,
       53.6
      ],
      [
       20.0,
       52.6
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       50.7
      ],
      [
       22.5,
       52.3
      ]
     ]
    },
    {
     "id": 28,
     "cx": 0.8,
     "cy": 0.3005,
     "w": 0.0278,
     "h": 0.0761,
     "mean": 40.1,
     "low_pct": 67.2,
     "cue": "reading / taking notes",
     "n": 201,
     "series": [
      [
       2.0,
       15.8
      ],
      [
       2.5,
       17.4
      ],
      [
       3.0,
       21.4
      ],
      [
       3.5,
       22.2
      ],
      [
       4.0,
       23.1
      ],
      [
       4.5,
       23.9
      ],
      [
       5.0,
       25.9
      ],
      [
       5.5,
       33.4
      ],
      [
       6.0,
       37.9
      ],
      [
       6.5,
       40.6
      ],
      [
       7.0,
       38.6
      ],
      [
       7.5,
       36.3
      ],
      [
       8.0,
       34.6
      ],
      [
       8.5,
       28.6
      ],
      [
       9.0,
       23.7
      ],
      [
       9.5,
       22.2
      ],
      [
       10.0,
       20.5
      ],
      [
       10.5,
       20.3
      ],
      [
       11.0,
       21.7
      ],
      [
       11.5,
       25.2
      ],
      [
       12.0,
       36.3
      ],
      [
       12.5,
       38.9
      ],
      [
       13.0,
       41.5
      ],
      [
       13.5,
       45.5
      ],
      [
       14.0,
       49.2
      ],
      [
       14.5,
       48.9
      ],
      [
       15.0,
       49.9
      ],
      [
       15.5,
       49.6
      ],
      [
       16.0,
       51.8
      ],
      [
       16.5,
       51.9
      ],
      [
       17.0,
       53.7
      ],
      [
       17.5,
       55.7
      ],
      [
       18.0,
       56.4
      ],
      [
       18.5,
       56.5
      ],
      [
       19.0,
       56.4
      ],
      [
       19.5,
       61.1
      ],
      [
       20.0,
       63.2
      ],
      [
       20.5,
       56.2
      ],
      [
       21.0,
       52.8
      ],
      [
       21.5,
       48.5
      ],
      [
       22.0,
       49.8
      ],
      [
       22.5,
       50.7
      ]
     ]
    },
    {
     "id": 17,
     "cx": 0.8779,
     "cy": 0.3438,
     "w": 0.0332,
     "h": 0.0792,
     "mean": 52.9,
     "low_pct": 4.9,
     "cue": "reading / taking notes",
     "n": 204,
     "series": [
      [
       2.0,
       50.0
      ],
      [
       2.5,
       48.1
      ],
      [
       3.0,
       51.1
      ],
      [
       3.5,
       48.7
      ],
      [
       4.0,
       51.6
      ],
      [
       4.5,
       52.3
      ],
      [
       5.0,
       52.7
      ],
      [
       5.5,
       53.1
      ],
      [
       6.0,
       52.9
      ],
      [
       6.5,
       55.9
      ],
      [
       7.0,
       58.4
      ],
      [
       7.5,
       59.8
      ],
      [
       8.0,
       59.7
      ],
      [
       8.5,
       53.0
      ],
      [
       9.0,
       52.9
      ],
      [
       9.5,
       53.1
      ],
      [
       10.0,
       56.8
      ],
      [
       10.5,
       54.8
      ],
      [
       11.0,
       52.2
      ],
      [
       11.5,
       52.6
      ],
      [
       12.0,
       53.6
      ],
      [
       12.5,
       55.1
      ],
      [
       13.0,
       53.7
      ],
      [
       13.5,
       53.6
      ],
      [
       14.0,
       56.0
      ],
      [
       14.5,
       53.3
      ],
      [
       15.0,
       54.3
      ],
      [
       15.5,
       53.2
      ],
      [
       16.0,
       56.5
      ],
      [
       16.5,
       57.0
      ],
      [
       17.0,
       53.2
      ],
      [
       17.5,
       50.0
      ],
      [
       18.0,
       50.0
      ],
      [
       18.5,
       50.0
      ],
      [
       19.0,
       50.0
      ],
      [
       19.5,
       50.0
      ],
      [
       20.0,
       50.0
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       50.0
      ],
      [
       21.5,
       50.0
      ],
      [
       22.0,
       50.0
      ],
      [
       22.5,
       50.0
      ]
     ]
    },
    {
     "id": 43,
     "cx": 0.8674,
     "cy": 0.4002,
     "w": 0.0416,
     "h": 0.0894,
     "mean": 55.9,
     "low_pct": 0.0,
     "cue": "reading / taking notes",
     "n": 197,
     "series": [
      [
       3.0,
       50.0
      ],
      [
       3.5,
       50.0
      ],
      [
       4.0,
       52.2
      ],
      [
       4.5,
       52.2
      ],
      [
       5.0,
       51.7
      ],
      [
       5.5,
       51.5
      ],
      [
       6.0,
       51.1
      ],
      [
       6.5,
       52.2
      ],
      [
       7.0,
       53.9
      ],
      [
       7.5,
       55.7
      ],
      [
       8.0,
       57.7
      ],
      [
       8.5,
       52.5
      ],
      [
       9.0,
       52.1
      ],
      [
       9.5,
       51.6
      ],
      [
       10.0,
       54.8
      ],
      [
       10.5,
       55.9
      ],
      [
       11.0,
       58.3
      ],
      [
       11.5,
       61.9
      ],
      [
       12.0,
       63.1
      ],
      [
       12.5,
       61.0
      ],
      [
       13.0,
       59.1
      ],
      [
       13.5,
       57.4
      ],
      [
       14.0,
       59.5
      ],
      [
       14.5,
       54.6
      ],
      [
       15.0,
       54.6
      ],
      [
       15.5,
       53.4
      ],
      [
       16.0,
       55.3
      ],
      [
       16.5,
       57.3
      ],
      [
       17.0,
       60.0
      ],
      [
       17.5,
       61.8
      ],
      [
       18.0,
       60.7
      ],
      [
       18.5,
       60.8
      ],
      [
       19.0,
       62.9
      ],
      [
       19.5,
       64.6
      ],
      [
       20.0,
       62.8
      ],
      [
       20.5,
       52.1
      ],
      [
       21.0,
       51.1
      ],
      [
       21.5,
       50.1
      ],
      [
       22.0,
       53.6
      ],
      [
       22.5,
       55.1
      ]
     ]
    },
    {
     "id": 42,
     "cx": 0.7881,
     "cy": 0.326,
     "w": 0.0434,
     "h": 0.1016,
     "mean": 13.6,
     "low_pct": 100.0,
     "cue": "oriented to lesson",
     "n": 2,
     "series": [
      [
       3.0,
       13.5
      ],
      [
       3.5,
       13.7
      ]
     ]
    },
    {
     "id": 16,
     "cx": 0.7595,
     "cy": 0.3461,
     "w": 0.0354,
     "h": 0.0893,
     "mean": 16.0,
     "low_pct": 100.0,
     "cue": "reading / writing",
     "n": 13,
     "series": [
      [
       3.0,
       14.5
      ],
      [
       3.5,
       14.4
      ],
      [
       4.0,
       16.8
      ],
      [
       4.5,
       18.5
      ]
     ]
    },
    {
     "id": 45,
     "cx": 0.7426,
     "cy": 0.2983,
     "w": 0.041,
     "h": 0.0832,
     "mean": 22.5,
     "low_pct": 100.0,
     "cue": "reading / writing",
     "n": 25,
     "series": [
      [
       3.5,
       19.1
      ],
      [
       4.0,
       18.2
      ],
      [
       4.5,
       18.3
      ],
      [
       5.0,
       21.6
      ],
      [
       7.0,
       26.2
      ],
      [
       7.5,
       31.5
      ]
     ]
    },
    {
     "id": 44,
     "cx": 0.7601,
     "cy": 0.3392,
     "w": 0.021,
     "h": 0.065,
     "mean": 28.2,
     "low_pct": 100.0,
     "cue": "reading / taking notes",
     "n": 17,
     "series": [
      [
       4.5,
       20.4
      ],
      [
       5.0,
       24.2
      ],
      [
       5.5,
       28.0
      ],
      [
       6.0,
       39.1
      ]
     ]
    },
    {
     "id": 46,
     "cx": 0.7455,
     "cy": 0.2997,
     "w": 0.0207,
     "h": 0.0602,
     "mean": 37.7,
     "low_pct": 89.9,
     "cue": "reading / writing",
     "n": 159,
     "series": [
      [
       6.5,
       40.0
      ],
      [
       7.0,
       39.7
      ],
      [
       7.5,
       38.4
      ],
      [
       8.0,
       36.4
      ],
      [
       8.5,
       29.0
      ],
      [
       9.0,
       22.0
      ],
      [
       9.5,
       20.4
      ],
      [
       10.0,
       18.7
      ],
      [
       10.5,
       16.2
      ],
      [
       11.0,
       16.5
      ],
      [
       11.5,
       17.5
      ],
      [
       12.0,
       27.1
      ],
      [
       12.5,
       40.1
      ],
      [
       13.0,
       41.4
      ],
      [
       13.5,
       39.1
      ],
      [
       14.0,
       36.8
      ],
      [
       14.5,
       37.2
      ],
      [
       15.0,
       39.8
      ],
      [
       15.5,
       47.0
      ],
      [
       16.0,
       48.4
      ],
      [
       16.5,
       45.2
      ],
      [
       17.0,
       42.7
      ],
      [
       17.5,
       39.9
      ],
      [
       18.0,
       38.9
      ],
      [
       18.5,
       35.8
      ],
      [
       19.0,
       37.8
      ],
      [
       19.5,
       40.8
      ],
      [
       20.0,
       52.7
      ],
      [
       20.5,
       50.0
      ],
      [
       21.0,
       49.7
      ],
      [
       21.5,
       48.7
      ],
      [
       22.0,
       53.8
      ],
      [
       22.5,
       55.9
      ]
     ]
    },
    {
     "id": 49,
     "cx": 0.7621,
     "cy": 0.3449,
     "w": 0.032,
     "h": 0.0795,
     "mean": 50.8,
     "low_pct": 3.4,
     "cue": "reading / taking notes",
     "n": 117,
     "series": [
      [
       11.0,
       41.7
      ],
      [
       11.5,
       30.9
      ],
      [
       12.0,
       50.0
      ],
      [
       12.5,
       50.0
      ],
      [
       13.0,
       50.0
      ],
      [
       13.5,
       50.0
      ],
      [
       14.0,
       50.0
      ],
      [
       14.5,
       50.0
      ],
      [
       15.0,
       50.0
      ],
      [
       15.5,
       50.0
      ],
      [
       16.0,
       52.4
      ],
      [
       16.5,
       52.8
      ],
      [
       17.0,
       52.6
      ],
      [
       17.5,
       51.1
      ],
      [
       18.0,
       50.0
      ],
      [
       18.5,
       50.0
      ],
      [
       19.0,
       51.1
      ],
      [
       19.5,
       57.6
      ],
      [
       20.0,
       61.4
      ],
      [
       20.5,
       53.6
      ],
      [
       21.0,
       52.0
      ],
      [
       21.5,
       50.3
      ],
      [
       22.0,
       54.7
      ],
      [
       22.5,
       56.7
      ]
     ]
    },
    {
     "id": 48,
     "cx": 0.7127,
     "cy": 0.2917,
     "w": 0.0488,
     "h": 0.1174,
     "mean": 21.9,
     "low_pct": 100.0,
     "cue": "oriented to lesson",
     "n": 19,
     "series": [
      [
       11.0,
       18.4
      ],
      [
       11.5,
       18.2
      ],
      [
       12.0,
       21.9
      ],
      [
       12.5,
       26.7
      ],
      [
       13.0,
       26.7
      ]
     ]
    },
    {
     "id": 50,
     "cx": 0.7992,
     "cy": 0.4144,
     "w": 0.0328,
     "h": 0.0776,
     "mean": 52.3,
     "low_pct": 10.5,
     "cue": "reading",
     "n": 105,
     "series": [
      [
       12.0,
       47.3
      ],
      [
       12.5,
       48.6
      ],
      [
       13.0,
       50.2
      ],
      [
       13.5,
       51.4
      ],
      [
       14.0,
       53.7
      ],
      [
       14.5,
       49.8
      ],
      [
       15.0,
       50.1
      ],
      [
       15.5,
       49.7
      ],
      [
       16.0,
       52.3
      ],
      [
       16.5,
       52.5
      ],
      [
       17.0,
       54.7
      ],
      [
       17.5,
       55.1
      ],
      [
       18.0,
       54.2
      ],
      [
       18.5,
       52.9
      ],
      [
       19.0,
       52.9
      ],
      [
       19.5,
       55.7
      ],
      [
       20.0,
       57.3
      ],
      [
       20.5,
       52.4
      ],
      [
       21.0,
       51.9
      ],
      [
       21.5,
       50.9
      ],
      [
       22.0,
       51.5
      ],
      [
       22.5,
       51.0
      ]
     ]
    },
    {
     "id": 51,
     "cx": 0.6974,
     "cy": 0.265,
     "w": 0.0411,
     "h": 0.0951,
     "mean": 28.2,
     "low_pct": 100.0,
     "cue": "reading / writing",
     "n": 69,
     "series": [
      [
       14.5,
       22.0
      ],
      [
       15.0,
       26.7
      ],
      [
       15.5,
       36.0
      ],
      [
       16.0,
       36.6
      ],
      [
       16.5,
       30.9
      ],
      [
       17.0,
       28.8
      ],
      [
       17.5,
       26.1
      ],
      [
       18.0,
       23.7
      ],
      [
       18.5,
       17.7
      ],
      [
       19.0,
       16.9
      ],
      [
       19.5,
       18.2
      ],
      [
       20.0,
       26.3
      ],
      [
       20.5,
       39.1
      ],
      [
       21.0,
       43.9
      ]
     ]
    },
    {
     "id": 52,
     "cx": 0.1669,
     "cy": 0.2753,
     "w": 0.0289,
     "h": 0.0817,
     "mean": 38.0,
     "low_pct": 94.3,
     "cue": "reading / taking notes",
     "n": 70,
     "series": [
      [
       15.5,
       27.6
      ],
      [
       16.0,
       32.4
      ],
      [
       16.5,
       41.0
      ],
      [
       17.0,
       41.4
      ],
      [
       17.5,
       38.4
      ],
      [
       18.0,
       37.4
      ],
      [
       18.5,
       38.6
      ],
      [
       19.0,
       35.7
      ],
      [
       19.5,
       34.9
      ],
      [
       20.0,
       39.4
      ],
      [
       20.5,
       39.2
      ],
      [
       21.0,
       38.6
      ],
      [
       21.5,
       38.6
      ],
      [
       22.0,
       39.8
      ],
      [
       22.5,
       42.4
      ]
     ]
    },
    {
     "id": 55,
     "cx": 0.6981,
     "cy": 0.3381,
     "w": 0.0275,
     "h": 0.0835,
     "mean": 48.3,
     "low_pct": 72.7,
     "cue": "reading / taking notes",
     "n": 11,
     "series": [
      [
       21.5,
       45.2
      ],
      [
       22.0,
       48.9
      ],
      [
       22.5,
       50.2
      ]
     ]
    }
   ],
   "class_mean": 48.69,
   "low_share": 23.55
  }
 ]
};
