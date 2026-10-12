# Digital Pathology Analysis Report

- **Source:** `pannuke_00003_256x256.png`
- **Image size:** 256 x 256 px
- **Model:** HoVer-Net (CoNSeP checkpoint), running on `cuda`
- **Analysed:** 2026-10-12T15:26:18+00:00 (UTC)

## Headline

- Nuclei detected: **11**
- Nuclear density: **671.4 nuclei/mm2** (field 0.0164 mm2)
- Scale: 0.5 um/px (assumed default)
- Runtime: 0.6 s over 9 tiles (9 processed)

## Cell type distribution

| Type | Colour | Count | Share | Mean area (px) | Mean confidence |
| --- | --- | ---: | ---: | ---: | ---: |
| Neoplastic | `#FF0000` | 10 | 90.9% | 1176.3 | 0.914 |
| Inflammatory | `#00C850` | 0 | 0.0% | 0.0 | 0 |
| Connective | `#005AFF` | 1 | 9.1% | 489.0 | 0.759 |
| Dead | `#FFC800` | 0 | 0.0% | 0.0 | 0 |
| Dead-karyotic | `#C800C8` | 0 | 0.0% | 0.0 | 0 |

## Nucleus size distribution

- Count: 11
- Mean: 1113.8 px, median: 995.0 px, sd: 513.0 px
- Range: 458 - 1909 px (p90: 1745 px)
- Class-confidence: mean 0.9, 100% above 0.5

## Largest nuclei

| id | type | area (px) | centroid (y, x) | confidence |
| ---: | --- | ---: | --- | ---: |
| 4 | Neoplastic | 1909 | (95, 95) | 0.55 |
| 1 | Neoplastic | 1745 | (200, 30) | 0.93 |
| 3 | Neoplastic | 1720 | (206, 97) | 0.99 |
| 7 | Neoplastic | 1636 | (198, 168) | 0.99 |
| 9 | Neoplastic | 1012 | (242, 182) | 0.97 |
| 6 | Neoplastic | 995 | (127, 139) | 0.99 |
| 8 | Neoplastic | 834 | (112, 172) | 0.94 |
| 2 | Neoplastic | 757 | (9, 52) | 0.99 |
| 11 | Neoplastic | 697 | (95, 246) | 0.98 |
| 5 | Connective | 489 | (50, 121) | 0.76 |

## Method

Images are split into 64-px-stride windows. Each 270x270 input patch yields an 80x80 prediction that is averaged where windows overlap, then nuclei are extracted with the HoVer-Net marker-controlled watershed (nuclear probability threshold + HoVer gradient maps) and labelled from the type head.

Parameters: tile 256 px, output 164 px, batch 8, min area 20 px, device cuda.

---

_Research and educational prototype. Not a medical device and not validated for clinical or diagnostic use._
