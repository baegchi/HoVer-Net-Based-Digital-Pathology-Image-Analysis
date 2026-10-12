# Digital Pathology Analysis Report

- **Source:** `TCGA-05-4245-01A-01-BS1.svs`
- **Image size:** 1536 x 1536 px
- **Model:** HoVer-Net (CoNSeP checkpoint), running on `cuda`
- **Analysed:** 2026-10-12T15:26:52+00:00 (UTC)

## Headline

- Nuclei detected: **944**
- Nuclear density: **1,590.9 nuclei/mm2** (field 0.5934 mm2)
- Scale: 0.5015 um/px (slide metadata)
- Runtime: 28.8 s over 529 tiles (529 processed)

## Cell type distribution

| Type | Colour | Count | Share | Mean area (px) | Mean confidence |
| --- | --- | ---: | ---: | ---: | ---: |
| Neoplastic | `#FF0000` | 138 | 14.6% | 220.1 | 0.555 |
| Inflammatory | `#00C850` | 54 | 5.7% | 129.9 | 0.403 |
| Connective | `#005AFF` | 187 | 19.8% | 151.2 | 0.678 |
| Dead | `#FFC800` | 536 | 56.8% | 93.1 | 0.849 |
| Dead-karyotic | `#C800C8` | 0 | 0.0% | 0.0 | 0 |

## Nucleus size distribution

- Count: 944
- Mean: 125.9 px, median: 99.5 px, sd: 90.6 px
- Range: 23 - 884 px (p90: 238 px)
- Class-confidence: mean 0.751, 79% above 0.5

## Largest nuclei

| id | type | area (px) | centroid (y, x) | confidence |
| ---: | --- | ---: | --- | ---: |
| 800 | Neoplastic | 884 | (1060, 454) | 0.93 |
| 35 | Neoplastic | 748 | (289, 67) | 0.95 |
| 910 | Connective | 642 | (1063, 1354) | 0.83 |
| 29 | Neoplastic | 564 | (657, 47) | 0.37 |
| 777 | Connective | 502 | (1565, 354) | 0.43 |
| 200 | Neoplastic | 491 | (311, 293) | 0.94 |
| 189 | Connective | 475 | (982, 276) | 0.87 |
| 107 | Neoplastic | 464 | (110, 172) | 0.90 |
| 638 | Neoplastic | 456 | (232, 1353) | 0.33 |
| 633 | Neoplastic | 454 | (63, 1318) | 0.65 |

## Method

Images are split into 64-px-stride windows. Each 270x270 input patch yields an 80x80 prediction that is averaged where windows overlap, then nuclei are extracted with the HoVer-Net marker-controlled watershed (nuclear probability threshold + HoVer gradient maps) and labelled from the type head.

Parameters: tile 256 px, output 164 px, batch 8, min area 20 px, device cuda.

---

_Research and educational prototype. Not a medical device and not validated for clinical or diagnostic use._
