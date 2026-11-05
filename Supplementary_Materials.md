# Online Supplementary Materials

**Manuscript Title:** Personalized yet the same? A multi-model audit of generative AI attraction recommendations across Chinese and international urban hierarchies  
**Target Journal:** *Telematics and Informatics* (Double-Anonymous Peer Review)

---

### Supplementary Table S1. Candidate 30-city stratification pool and composite open-encyclopedic information index ($Z_{\\text{info}}$)

To avoid ad-hoc destination selection, 15 Chinese tourist cities and 15 international tourist cities were queried live against the Overpass OpenStreetMap API (municipal `tourism=*` nodes), the Wikimedia Pageviews REST API (2025 annual desktop + mobile views summed across English and primary local Wikipedia editions), and the Wikidata Action API (city-entity sitelinks). Within each 15-city origin bloc, log-transformed indicators were standardized into an equal-weight composite index $Z_{\\text{info}} = \\frac{1}{3}(z_{\\ln \\text{OSM}} + z_{\\ln \\text{WikiViews}} + z_{\\ln \\text{Sitelinks}})$. Six Info-Rich metropolises (`60.0%–100.0%` within-bloc percentile) and four Info-Scarce secondary cultural cities (`6.7%–33.3%` within-bloc percentile) were selected for the live API audit.

| City ID | City | Country / Region | Origin Bloc | OSM Tourism Nodes | 2025 Wiki Views (k/yr) | Wikidata Sitelinks | Composite Z_info | Bloc Percentile (%) | Audit Sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BJS | Beijing | China | Domestic | 4,820 | 1,487.4 | 258 | +1.372 | 100.0% | Selected: Info-Rich |
| SHA | Shanghai | China | Domestic | 4,310 | 1,729.4 | 211 | +1.251 | 93.3% | Selected: Info-Rich |
| CTU | Chengdu | China | Domestic | 3,190 | 827.0 | 114 | +0.513 | 73.3% | Selected: Info-Rich |
| CAN | Guangzhou | China | Domestic | 3,580 | 1,238.9 | 145 | +0.838 | 86.7% | Unselected Candidate (Rich Stratum) |
| HGH | Hangzhou | China | Domestic | 3,490 | 614.1 | 113 | +0.448 | 53.3% | Unselected Candidate (Middle Buffer) |
| SIA | Xi'an | China | Domestic | 3,120 | 662.8 | 121 | +0.469 | 66.7% | Unselected Candidate (Rich Stratum) |
| CKG | Chongqing | China | Domestic | 2,980 | 1,798.8 | 133 | +0.830 | 80.0% | Unselected Candidate (Rich Stratum) |
| NKG | Nanjing | China | Domestic | 2,490 | 754.5 | 125 | +0.439 | 46.7% | Unselected Candidate (Middle Buffer) |
| WUH | Wuhan | China | Domestic | 2,240 | 539.1 | 164 | +0.454 | 60.0% | Unselected Candidate (Rich Stratum) |
| XMN | Xiamen | China | Domestic | 1,850 | 380.5 | 75 | -0.216 | 40.0% | Unselected Candidate (Middle Buffer) |
| QZJ | Quanzhou | China | Domestic | 680 | 198.5 | 62 | -0.951 | 33.3% | Selected: Info-Scarce |
| DLU | Dali | China | Domestic | 540 | 118.6 | 41 | -1.465 | 13.3% | Selected: Info-Scarce |
| CHZ | Chaozhou | China | Domestic | 590 | 138.8 | 57 | -1.176 | 26.7% | Unselected Candidate (Scarce Stratum) |
| JDZ | Jingdezhen | China | Domestic | 510 | 77.0 | 46 | -1.557 | 6.7% | Unselected Candidate (Scarce Stratum) |
| DTG | Datong | China | Domestic | 620 | 110.9 | 55 | -1.250 | 20.0% | Unselected Candidate (Scarce Stratum) |
| PAR | Paris | France | International | 6,450 | 2,951.0 | 289 | +0.946 | 93.3% | Selected: Info-Rich |
| TYO | Tokyo | Japan | International | 5,890 | 3,118.6 | 267 | +0.867 | 73.3% | Selected: Info-Rich |
| BCN | Barcelona | Spain | International | 4,120 | 1,854.4 | 202 | +0.277 | 60.0% | Selected: Info-Rich |
| LON | London | United Kingdom | International | 6,120 | 3,168.9 | 284 | +0.940 | 86.7% | Unselected Candidate (Rich Stratum) |
| NYC | New York | United States | International | 5,520 | 5,528.9 | 264 | +1.066 | 100.0% | Unselected Candidate (Rich Stratum) |
| ROM | Rome | Italy | International | 4,980 | 2,471.0 | 273 | +0.718 | 66.7% | Unselected Candidate (Rich Stratum) |
| SIN | Singapore | Singapore | International | 3,820 | 5,648.8 | 270 | +0.938 | 80.0% | Unselected Candidate (Rich Stratum) |
| SEL | Seoul | South Korea | International | 3,250 | 1,155.6 | 222 | +0.058 | 40.0% | Unselected Candidate (Middle Buffer) |
| VIE | Vienna | Austria | International | 2,720 | 1,948.6 | 242 | +0.267 | 53.3% | Unselected Candidate (Middle Buffer) |
| PRG | Prague | Czechia | International | 2,440 | 1,641.3 | 222 | +0.081 | 46.7% | Unselected Candidate (Middle Buffer) |
| OPO | Porto | Portugal | International | 920 | 637.8 | 119 | -1.224 | 20.0% | Selected: Info-Scarce |
| CNX | Chiang Mai | Thailand | International | 810 | 411.1 | 76 | -1.820 | 6.7% | Selected: Info-Scarce |
| KYO | Kyoto | Japan | International | 1,180 | 947.8 | 231 | -0.420 | 33.3% | Unselected Candidate (Scarce Stratum) |
| BRU | Bruges | Belgium | International | 790 | 537.9 | 100 | -1.499 | 13.3% | Unselected Candidate (Scarce Stratum) |
| GRX | Granada | Spain | International | 860 | 624.6 | 129 | -1.196 | 26.7% | Unselected Candidate (Scarce Stratum) |

---

### Supplementary Table S2. Nine-category POI sampling frame, 3-tier physical accessibility rubric, and bilingual constraint-cue dictionaries

#### S2A. Category distribution across the 600-attraction municipal ground-truth catalog

| Attraction Category | Total POIs (N=600) | Info-Rich (6 Cities) | Info-Scarce (4 Cities) | Step-Free Tier 2 (%) | Mean Dist. to Center (km) |
| --- | --- | --- | --- | --- | --- |
| Art District | 43 | 31 | 12 | 69.8% | 7.09 |
| Cultural District | 87 | 51 | 36 | 42.5% | 8.80 |
| Heritage Park | 59 | 46 | 13 | 67.8% | 7.93 |
| Historic | 102 | 54 | 48 | 28.4% | 17.77 |
| Modern Landmark | 41 | 33 | 8 | 53.7% | 5.47 |
| Museum | 76 | 59 | 17 | 73.7% | 9.64 |
| Nature | 69 | 23 | 46 | 29.0% | 25.12 |
| Religious | 96 | 42 | 54 | 33.3% | 8.00 |
| Theme Park | 27 | 21 | 6 | 66.7% | 19.48 |

#### S2B. Three-tier physical step-free accessibility coding rubric (`wheelchair_accessible`)

| Accessibility Tier | Binary Code in Models | Operational Verification Criteria | Share of 600-POI Catalog |
| --- | --- | --- | --- |
| **Tier 2: Full Step-Free Access** | `1` | Continuous step-free entrance and primary visitor circulation confirmed via OpenStreetMap `wheelchair=yes` tags, official venue barrier-free visitor guides, and municipal accessibility directories (ramps/elevators across main halls/paths, accessible restrooms). | `47.3%` (`59.2%` Info-Rich vs. `29.6%` Info-Scarce) |
| **Tier 1: Partial / Assisted Access** | `0` | Ground-level courtyard or partial gallery access only; steep thresholds, gravel/cobblestone alleys, or secondary pavilions requiring manual assistance (`wheelchair=limited`). | Combined `52.7%` (`40.8%` Info-Rich vs. `70.4%` Info-Scarce) |
| **Tier 0: Stepped / Inaccessible** | `0` | Unretrofitted multi-flight staircases, steep mountain trails, cliffside temples, or historic courtyard residences lacking step-free ramps or lifts (`wheelchair=no`). | Combined with Tier 1 above |

#### S2C. Bilingual regex cue dictionaries for discursive tailoring audit

- **Accessibility / Step-Free Cues (`ACC_CUE_RE`):** `(无障碍|轮椅|平缓|电梯|坡道|平坦|免爬坡|步道平|step-free|wheelchair|accessible|ramp|elevator|flat|barrier-free|level path)` — Note: generic terrain descriptors (`flat`, `paved`, `平缓`, `平坦`) appear naturally in `42.2%` of unrestricted park/plaza descriptions, providing the baseline against which `Wheelchair` personas jump by `+45.1 pp` to `87.2%`.
- **Companion / Age-Group Cues (`FAM_CUE_RE`):** `(亲子|儿童|孩子|老人|长辈|银发|慢节奏|科普|互动|休息|family|child|kid|senior|elderly|older|gentle pace|educational|interactive|seating|rest area)`
- **Budget / Cost Cues (`BUD_CUE_RE`):** `(免费|免票|低成本|穷游|性价比|门票便宜|高端|私享|尊享|品质|奢华|定制|free|budget|affordable|low-cost|no admission|luxury|exclusive|premium|upscale|private|bespoke)`

---

### Supplementary Table S3. Full 25-column group-level diversity, concentration, rank-overlap, and discursive metrics

*Definitions of supplementary benchmark alignment columns:* `Public_Top10_RBO` measures Rank-Biased Overlap ($p = 0.85, k = 10$) between each endpoint's aggregated mention-frequency top-10 ranking in city $c$ and the exogenous top-10 ranking by 2025 Wikimedia annual pageviews. `JSD_Salience` measures the base-2 Jensen–Shannon divergence $\\text{JSD}(P_{\\text{AI}} \\parallel P_{\\text{Wiki}}) \\in [0, 1]$ between normalized catalog mention shares and normalized 2025 Wikimedia pageview shares across all 60 cataloged POIs in city $c$.

| model | origin | richness | N0_unique | N1_effective | HHI | Gini_OSM | Gini_Sample | Gini_EqualCap_M15 | Gini_AllOSM | Top10_share_pct | Noise_RBO | Noise_RBO_SD | Between_Ctx_RBO | NPI_RBO | NPI_CI_Low | NPI_CI_High | NNPI_RBO | Delta_K10_Slots | Discursive_NNPI | Blank_B2_RBO | Blank_B2_Shared_K10 | Human_Top10_RBO | JSD_vs_Reviews | Hallucination_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.6-sol | Domestic | Info-Rich | 38.67 | 20.89 | 0.0630 | 0.7370 | 0.5910 | 0.6330 | 0.9960 | 70.72 | 0.7190 | 0.1420 | 0.4040 | 0.3150 | 0.2650 | 0.3640 | 0.4380 | 2.3550 | 0.5490 | 0.4410 | 4.8710 | 0.2270 | 0.5170 | 0.9000 |
| claude-sonnet-4-6 | Domestic | Info-Rich | 36.33 | 20.27 | 0.0630 | 0.7450 | 0.5790 | 0.6240 | 0.9960 | 71.08 | 0.7160 | 0.1390 | 0.5250 | 0.1910 | 0.1440 | 0.2410 | 0.2670 | 1.8650 | 0.4760 | 0.5160 | 5.1820 | 0.2780 | 0.4940 | 2.0130 |
| deepseek-v4-pro-0813 | Domestic | Info-Rich | 36.67 | 20.38 | 0.0610 | 0.7420 | 0.5760 | 0.6220 | 0.9960 | 70.21 | 0.7460 | 0.1210 | 0.4560 | 0.2900 | 0.2390 | 0.3400 | 0.3890 | 2.6310 | 0.4570 | 0.4730 | 4.9170 | 0.3220 | 0.4740 | 2.0800 |
| qwen3.7-flash | Domestic | Info-Rich | 45.33 | 24.33 | 0.0540 | 0.6930 | 0.5950 | 0.6000 | 0.9950 | 64.72 | 0.5160 | 0.1670 | 0.3030 | 0.2120 | 0.1620 | 0.2580 | 0.4130 | 1.5890 | 0.3920 | 0.3660 | 3.7560 | 0.1860 | 0.5270 | 7.2230 |
| gpt-5.6-sol | Domestic | Info-Scarce | 33.00 | 18.35 | 0.0670 | 0.7660 | 0.5690 | 0.6460 | 0.9770 | 72.58 | 0.7310 | 0.1270 | 0.4910 | 0.2400 | 0.2030 | 0.2760 | 0.3270 | 1.5960 | 0.5220 | 0.4870 | 5.4050 | 0.1930 | 0.4610 | 1.0400 |
| claude-sonnet-4-6 | Domestic | Info-Scarce | 34.50 | 18.66 | 0.0680 | 0.7660 | 0.5850 | 0.6250 | 0.9770 | 74.50 | 0.6930 | 0.1190 | 0.5850 | 0.1080 | 0.0720 | 0.1430 | 0.1550 | 1.0220 | 0.4100 | 0.5940 | 6.0080 | 0.2400 | 0.4260 | 2.0450 |
| deepseek-v4-pro-0813 | Domestic | Info-Scarce | 29.00 | 18.14 | 0.0690 | 0.7710 | 0.5220 | 0.6470 | 0.9770 | 74.78 | 0.7470 | 0.1160 | 0.5520 | 0.1950 | 0.1630 | 0.2270 | 0.2610 | 1.6930 | 0.4660 | 0.5900 | 5.6290 | 0.2290 | 0.4390 | 0.9750 |
| qwen3.7-flash | Domestic | Info-Scarce | 34.50 | 18.65 | 0.0710 | 0.7670 | 0.5900 | 0.6530 | 0.9770 | 73.89 | 0.5860 | 0.1570 | 0.4690 | 0.1170 | 0.0810 | 0.1530 | 0.2050 | 0.8230 | 0.3980 | 0.5220 | 4.5420 | 0.2200 | 0.4100 | 8.3350 |
| gpt-5.6-sol | International | Info-Rich | 40.33 | 24.22 | 0.0510 | 0.6930 | 0.5430 | 0.6210 | 0.9970 | 62.82 | 0.7100 | 0.1550 | 0.3010 | 0.4080 | 0.3530 | 0.4590 | 0.5760 | 3.2910 | 0.4710 | 0.3890 | 4.3780 | 0.2580 | 0.3580 | 4.2130 |
| claude-sonnet-4-6 | International | Info-Rich | 42.33 | 23.55 | 0.0540 | 0.7040 | 0.5790 | 0.6190 | 0.9970 | 63.77 | 0.6500 | 0.1710 | 0.3730 | 0.2770 | 0.2270 | 0.3260 | 0.4310 | 2.5460 | 0.4090 | 0.4620 | 4.7550 | 0.2860 | 0.3490 | 5.5900 |
| deepseek-v4-pro-0813 | International | Info-Rich | 40.67 | 23.50 | 0.0540 | 0.7050 | 0.5620 | 0.5940 | 0.9970 | 64.99 | 0.7340 | 0.1570 | 0.3940 | 0.3400 | 0.2860 | 0.3920 | 0.4640 | 3.3070 | 0.4530 | 0.4870 | 4.7270 | 0.3570 | 0.3180 | 4.8400 |
| qwen3.7-flash | International | Info-Rich | 49.00 | 31.46 | 0.0390 | 0.6010 | 0.5110 | 0.5430 | 0.9950 | 52.16 | 0.4920 | 0.1710 | 0.2020 | 0.2900 | 0.2420 | 0.3360 | 0.5920 | 2.4120 | 0.3550 | 0.2360 | 3.0990 | 0.1710 | 0.3530 | 10.05 |
| gpt-5.6-sol | International | Info-Scarce | 38.00 | 25.18 | 0.0470 | 0.6740 | 0.4840 | 0.6040 | 0.9770 | 59.02 | 0.6280 | 0.1510 | 0.2400 | 0.3890 | 0.3400 | 0.4340 | 0.6180 | 2.9480 | 0.4340 | 0.2580 | 3.6720 | 0.1740 | 0.3910 | 3.1950 |
| claude-sonnet-4-6 | International | Info-Scarce | 38.00 | 23.70 | 0.0510 | 0.6950 | 0.5180 | 0.6140 | 0.9790 | 61.56 | 0.5570 | 0.1840 | 0.3110 | 0.2450 | 0.1940 | 0.2970 | 0.4410 | 2.0900 | 0.3630 | 0.3700 | 3.9960 | 0.2110 | 0.3860 | 7.6000 |
| deepseek-v4-pro-0813 | International | Info-Scarce | 36.00 | 23.39 | 0.0520 | 0.7010 | 0.4990 | 0.6180 | 0.9790 | 62.66 | 0.6440 | 0.1570 | 0.3380 | 0.3060 | 0.2590 | 0.3500 | 0.4730 | 2.7980 | 0.3990 | 0.4390 | 4.6980 | 0.2600 | 0.3620 | 4.9300 |
| qwen3.7-flash | International | Info-Scarce | 32.50 | 21.45 | 0.0560 | 0.7260 | 0.4950 | 0.6460 | 0.9810 | 66.82 | 0.3710 | 0.1690 | 0.2070 | 0.1640 | 0.1290 | 0.1970 | 0.4610 | 1.2620 | 0.3370 | 0.2960 | 3.2700 | 0.1720 | 0.3790 | 16.14 |

---

### Supplementary Table S4. Commercial LLM API endpoint routing, snapshot aliases, and valid completion log

| endpoint_alias | api_returned_model | model_origin | provider | valid_queries_n | mean_latency_sec | mean_completion_tokens | sha256_verified_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claude-sonnet-4-6 | claude-sonnet-4-6 | International | relay | 2541 | 20.42 | 0.0000 | 100.00 |
| deepseek-v4-pro-0813 | deepseek-v4-pro-0813 | Domestic | pomo | 1424 | 73.96 | 4597.80 | 100.00 |
| deepseek-v4-pro-0813 | deepseek-v4-pro-ga-260813 | Domestic | pomo | 1126 | 72.06 | 4535.00 | 100.00 |
| gpt-5.6-sol | gpt-5.6-sol | International | pomo | 2550 | 36.58 | 1071.70 | 100.00 |
| qwen3.7-flash | qwen3.7-flash | Domestic | pomo | 2550 | 48.77 | 4671.10 | 100.00 |

---

### Supplementary Table S5. Rank-stratified set overlap (`Top-1` to `Top-10`) and matched 12-context `Batch B1` ($\\tau = 0.7$) vs. `Batch B5` ($\\tau = 0.0$) comparison

#### S5A. Rank-depth set Jaccard overlap and net slot turnover in `Batch B1` ($N = 5,752$ queries)

| Depth_k | k | Within_Jaccard_at_k | Between_Jaccard_at_k | Jaccard_Ratio_Bet_over_Within | Within_SlotAgreement_at_k | Between_SlotAgreement_at_k | Net_Slot_Turnover_at_k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Top-1 | 1 | 0.6156 | 0.3429 | 0.5571 | 0.6156 | 0.3429 | 0.2730 |
| Top-3 | 3 | 0.5426 | 0.2781 | 0.5126 | 0.6471 | 0.3724 | 0.8240 |
| Top-5 | 5 | 0.5364 | 0.2831 | 0.5278 | 0.6595 | 0.3965 | 1.3150 |
| Top-10 | 10 | 0.5152 | 0.2995 | 0.5813 | 0.6515 | 0.4304 | 2.2110 |

#### S5B. Matched 12-persona (`Peak × Unrestricted`) comparison between stochastic `Batch B1` ($\\tau = 0.7$) and greedy `Batch B5` ($\\tau = 0.0$)

| Regime | Contexts_Evaluated | Replicates_Per_Cell | Within_Noise_RBO | Between_Persona_RBO | Between_Persona_Jaccard | Shared_POIs_Out_Of_10 | NPI_RBO | NNPI_RBO | Delta_K10_Slots | Mean_Gini_OSM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Batch_B1_Full_48Contexts (tau=0.7, N=5,752) | 48 | 3 | 0.6440 | 0.3816 | 0.2995 | 4.3000 | 0.2625 | 0.4148 | 2.2100 | 0.7149 |
| Batch_B1_Matched_12Contexts (tau=0.7, N=1,438) | 12 | 3 | 0.7022 | 0.4625 | 0.3508 | 4.9000 | 0.2397 | 0.3469 | 2.0600 | 0.7414 |
| Batch_B5_Greedy_12Contexts (tau=0.0, N=1,440) | 12 | 3 | 0.7202 | 0.4667 | 0.3549 | 4.9400 | 0.2535 | 0.3566 | 2.0600 | 0.7463 |

---

### Supplementary Table S6. Four-endpoint collective monoculture omission audit and out-of-catalog municipal (`Class B1`) recall

#### S6A. Simultaneous four-endpoint zero-visibility vs. marginal independence (2,000 column permutations per city)

| city_id | city | origin | richness | Mean_Single_ZeroVis_pct | All4_Omitted_Count | All4_Omitted_pct | Expected_Indep_pct | Perm_Null_Mean_pct | Excess_Omission_pp | Excess_Ratio_Obs_over_Exp | Perm_p_val |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BJS | Beijing | Domestic | Info-Rich | 39.17 | 10 | 16.67 | 2.1700 | 2.1600 | 14.50 | 7.6900 | 0.0005 |
| SHA | Shanghai | Domestic | Info-Rich | 24.58 | 7 | 11.67 | 0.3500 | 0.3400 | 11.31 | 33.09 | 0.0005 |
| CTU | Chengdu | Domestic | Info-Rich | 40.00 | 13 | 21.67 | 2.3300 | 2.3700 | 19.33 | 9.2900 | 0.0005 |
| QZJ | Quanzhou | Domestic | Info-Scarce | 50.42 | 22 | 36.67 | 6.4200 | 6.4700 | 30.25 | 5.7100 | 0.0005 |
| DLU | Dali | Domestic | Info-Scarce | 40.42 | 14 | 23.33 | 2.5900 | 2.5500 | 20.74 | 9.0000 | 0.0005 |
| PAR | Paris | International | Info-Rich | 27.50 | 4 | 6.6700 | 0.4400 | 0.4600 | 6.2200 | 15.04 | 0.0005 |
| TYO | Tokyo | International | Info-Rich | 24.58 | 3 | 5.0000 | 0.3400 | 0.3500 | 4.6600 | 14.61 | 0.0015 |
| BCN | Barcelona | International | Info-Rich | 32.50 | 12 | 20.00 | 1.0400 | 1.1000 | 18.96 | 19.29 | 0.0005 |
| OPO | Porto | International | Info-Scarce | 45.83 | 23 | 38.33 | 4.4000 | 4.4300 | 33.94 | 8.7200 | 0.0005 |
| CNX | Chiang Mai | International | Info-Scarce | 33.75 | 10 | 16.67 | 1.2100 | 1.1700 | 15.46 | 13.76 | 0.0005 |

#### S6B. Out-of-catalog verified municipal POI (`Class B1`) slot share and unique venue counts by city

| city_id | city | richness | model | InCatalog_Unique_N0 | OutOfCatalog_Municipal_Unique | Total_Municipal_Unique_All | OutOfCatalog_Slot_Share_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BJS | Beijing | Info-Rich | gpt-5.6-sol | 35 | 16 | 51 | 14.51 |
| BJS | Beijing | Info-Rich | claude-sonnet-4-6 | 33 | 14 | 47 | 11.46 |
| BJS | Beijing | Info-Rich | deepseek-v4-pro-0813 | 34 | 13 | 47 | 11.18 |
| BJS | Beijing | Info-Rich | qwen3.7-flash | 44 | 32 | 76 | 14.93 |
| SHA | Shanghai | Info-Rich | gpt-5.6-sol | 46 | 24 | 70 | 13.96 |
| SHA | Shanghai | Info-Rich | claude-sonnet-4-6 | 44 | 18 | 62 | 7.2900 |
| SHA | Shanghai | Info-Rich | deepseek-v4-pro-0813 | 43 | 32 | 75 | 11.11 |
| SHA | Shanghai | Info-Rich | qwen3.7-flash | 48 | 32 | 80 | 14.31 |
| CTU | Chengdu | Info-Rich | gpt-5.6-sol | 35 | 21 | 56 | 6.6000 |
| CTU | Chengdu | Info-Rich | claude-sonnet-4-6 | 32 | 9 | 41 | 3.8200 |
| CTU | Chengdu | Info-Rich | deepseek-v4-pro-0813 | 33 | 11 | 44 | 2.1500 |
| CTU | Chengdu | Info-Rich | qwen3.7-flash | 44 | 30 | 74 | 6.1100 |
| QZJ | Quanzhou | Info-Scarce | gpt-5.6-sol | 30 | 40 | 70 | 10.83 |
| QZJ | Quanzhou | Info-Scarce | claude-sonnet-4-6 | 30 | 13 | 43 | 1.1100 |
| QZJ | Quanzhou | Info-Scarce | deepseek-v4-pro-0813 | 27 | 20 | 47 | 3.4700 |
| QZJ | Quanzhou | Info-Scarce | qwen3.7-flash | 32 | 101 | 133 | 12.85 |
| DLU | Dali | Info-Scarce | gpt-5.6-sol | 36 | 22 | 58 | 3.0600 |
| DLU | Dali | Info-Scarce | claude-sonnet-4-6 | 39 | 20 | 59 | 3.0600 |
| DLU | Dali | Info-Scarce | deepseek-v4-pro-0813 | 31 | 17 | 48 | 4.3100 |
| DLU | Dali | Info-Scarce | qwen3.7-flash | 37 | 42 | 79 | 5.0000 |
| PAR | Paris | Info-Rich | gpt-5.6-sol | 39 | 15 | 54 | 8.1900 |
| PAR | Paris | Info-Rich | claude-sonnet-4-6 | 41 | 10 | 51 | 4.2400 |
| PAR | Paris | Info-Rich | deepseek-v4-pro-0813 | 42 | 10 | 52 | 5.0700 |
| PAR | Paris | Info-Rich | qwen3.7-flash | 52 | 20 | 72 | 4.5800 |
| TYO | Tokyo | Info-Rich | gpt-5.6-sol | 42 | 26 | 68 | 11.60 |
| TYO | Tokyo | Info-Rich | claude-sonnet-4-6 | 46 | 32 | 78 | 6.9400 |
| TYO | Tokyo | Info-Rich | deepseek-v4-pro-0813 | 44 | 17 | 61 | 8.2600 |
| TYO | Tokyo | Info-Rich | qwen3.7-flash | 49 | 52 | 101 | 12.78 |
| BCN | Barcelona | Info-Rich | gpt-5.6-sol | 40 | 16 | 56 | 3.7500 |
| BCN | Barcelona | Info-Rich | claude-sonnet-4-6 | 40 | 11 | 51 | 2.9900 |
| BCN | Barcelona | Info-Rich | deepseek-v4-pro-0813 | 36 | 7 | 43 | 2.5000 |
| BCN | Barcelona | Info-Rich | qwen3.7-flash | 46 | 49 | 95 | 6.9400 |
| OPO | Porto | Info-Scarce | gpt-5.6-sol | 33 | 22 | 55 | 5.5600 |
| OPO | Porto | Info-Scarce | claude-sonnet-4-6 | 34 | 16 | 50 | 2.0100 |
| OPO | Porto | Info-Scarce | deepseek-v4-pro-0813 | 31 | 11 | 42 | 2.0100 |
| OPO | Porto | Info-Scarce | qwen3.7-flash | 32 | 48 | 80 | 5.9000 |
| CNX | Chiang Mai | Info-Scarce | gpt-5.6-sol | 43 | 7 | 50 | 5.3500 |
| CNX | Chiang Mai | Info-Scarce | claude-sonnet-4-6 | 42 | 27 | 69 | 4.8300 |
| CNX | Chiang Mai | Info-Scarce | deepseek-v4-pro-0813 | 41 | 9 | 50 | 1.5300 |
| CNX | Chiang Mai | Info-Scarce | qwen3.7-flash | 33 | 162 | 195 | 20.62 |

---

### Supplementary Table S7. Comparison of REML Model SEs, One-Way List SEs, Symmetric Two-Way Dyadic SEs, and Hierarchical City-Cluster SEs ($df = 9$) on the $N = 9,600$ dyadic panel

Because every dyad pairs two recommendation lists within the same city $c$, clustering marginal composite residuals at the city level ($G = 10$, evaluated against $t_{df=9}$) absorbs both symmetric two-way dyadic list sharing and city-level random-intercept covariance, yielding the most conservative standard errors for city-varying predictors (`Home-Field Match`, `Info-Rich Stratum`, and their interaction).

| Outcome | Predictor | Coef | REML Model SE | 1-Way List SE | 2-Way Dyadic SE | City-Cluster SE (G=10) | t-stat (df=9) | p-val (City df=9) | BH-FDR q (m=9) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RBO (p=0.85) | Intercept (Baseline Overlap) | +0.5299 | 0.0460 | 0.0110 | 0.0141 | 0.0628 | +8.43 | <0.0001 | - |
| RBO (p=0.85) | Different Companion | -0.0707 | 0.0058 | 0.0073 | 0.0078 | 0.0113 | -6.27 | 0.0001 | 0.0006 |
| RBO (p=0.85) | Different Budget | -0.1193 | 0.0058 | 0.0075 | 0.0083 | 0.0196 | -6.07 | 0.0002 | 0.0006 |
| RBO (p=0.85) | Different Season | -0.0172 | 0.0059 | 0.0071 | 0.0073 | 0.0072 | -2.38 | 0.0412 | 0.0742 |
| RBO (p=0.85) | Different Mobility | -0.1103 | 0.0059 | 0.0075 | 0.0087 | 0.0209 | -5.27 | 0.0005 | 0.0012 |
| RBO (p=0.85) | Same Endpoint-Tier Configuration | +0.0875 | 0.0063 | 0.0114 | 0.0125 | 0.0080 | +10.99 | <0.0001 | <0.0001 |
| RBO (p=0.85) | Same Geo-Bloc / Origin | -0.0024 | 0.0072 | 0.0133 | 0.0169 | 0.0165 | -0.15 | 0.8857 | 0.9388 |
| RBO (p=0.85) | Home-Field Match | -0.0214 | 0.0079 | 0.0171 | 0.0218 | 0.0442 | -0.48 | 0.6407 | 0.9388 |
| RBO (p=0.85) | Info-Rich Stratum | -0.0061 | 0.0592 | 0.0123 | 0.0167 | 0.0774 | -0.08 | 0.9388 | 0.9388 |
| RBO (p=0.85) | Home-Field Match x Info-Rich | -0.0046 | 0.0096 | 0.0195 | 0.0242 | 0.0570 | -0.08 | 0.9371 | 0.9388 |
| Jaccard@10 | Intercept (Baseline Overlap) | +0.4229 | 0.0355 | 0.0076 | 0.0100 | 0.0422 | +10.02 | <0.0001 | - |
| Jaccard@10 | Different Companion | -0.0670 | 0.0043 | 0.0050 | 0.0053 | 0.0088 | -7.59 | <0.0001 | 0.0001 |
| Jaccard@10 | Different Budget | -0.0866 | 0.0043 | 0.0050 | 0.0055 | 0.0137 | -6.33 | 0.0001 | 0.0003 |
| Jaccard@10 | Different Season | -0.0160 | 0.0044 | 0.0048 | 0.0047 | 0.0044 | -3.65 | 0.0053 | 0.0096 |
| Jaccard@10 | Different Mobility | -0.1012 | 0.0044 | 0.0048 | 0.0054 | 0.0109 | -9.31 | <0.0001 | <0.0001 |
| Jaccard@10 | Same Endpoint-Tier Configuration | +0.0808 | 0.0047 | 0.0088 | 0.0099 | 0.0057 | +14.19 | <0.0001 | <0.0001 |
| Jaccard@10 | Same Geo-Bloc / Origin | -0.0030 | 0.0054 | 0.0097 | 0.0125 | 0.0141 | -0.21 | 0.8383 | 0.9441 |
| Jaccard@10 | Home-Field Match | -0.0103 | 0.0059 | 0.0134 | 0.0167 | 0.0461 | -0.22 | 0.8290 | 0.9441 |
| Jaccard@10 | Info-Rich Stratum | -0.0082 | 0.0457 | 0.0092 | 0.0125 | 0.0583 | -0.14 | 0.8919 | 0.9441 |
| Jaccard@10 | Home-Field Match x Info-Rich | +0.0036 | 0.0072 | 0.0156 | 0.0187 | 0.0504 | +0.07 | 0.9441 | 0.9441 |

---

### Supplementary Table S8. Empirical propensity Maximal Marginal Relevance (MMR) catalog post-processing simulation (`Beijing` vs. `Quanzhou`)

#### S8A. City-by-endpoint baseline vs. spatial MMR ($\lambda_2 = 0.25, \lambda_3 = 0.20$) across 48 personas

| City | Model | Regime | Gini_OSM | N0_unique_pois | Zero_visibility_pct | Tail_Share_pct | ROSR_tail | NPI_RBO | SDE_Change_vs_OSM_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Beijing (Info-Rich) | gpt-5.6-sol | 1_Empirical_Baseline_BatchB1 | 0.7829 | 35.00 | 41.67 | 26.32 | 0.4510 | 0.3297 | -38.84 |
| Beijing (Info-Rich) | gpt-5.6-sol | 2_Catalog_MMR_PostProcessing_Sim (lam_geo=0.25, lam_tail=0.20) | 0.6669 | 54.00 | 10.00 | 38.61 | 0.6620 | 0.2675 | -23.38 |
| Beijing (Info-Rich) | claude-sonnet-4-6 | 1_Empirical_Baseline_BatchB1 | 0.7765 | 33.00 | 45.00 | 25.77 | 0.4420 | 0.1975 | -35.96 |
| Beijing (Info-Rich) | claude-sonnet-4-6 | 2_Catalog_MMR_PostProcessing_Sim (lam_geo=0.25, lam_tail=0.20) | 0.6370 | 53.00 | 11.67 | 38.19 | 0.6550 | 0.1354 | -21.64 |
| Beijing (Info-Rich) | deepseek-v4-pro-0813 | 1_Empirical_Baseline_BatchB1 | 0.7837 | 34.00 | 43.33 | 22.43 | 0.3850 | 0.2329 | -38.62 |
| Beijing (Info-Rich) | deepseek-v4-pro-0813 | 2_Catalog_MMR_PostProcessing_Sim (lam_geo=0.25, lam_tail=0.20) | 0.6440 | 54.00 | 10.00 | 36.46 | 0.6250 | 0.1789 | -22.94 |
| Beijing (Info-Rich) | qwen3.7-flash | 1_Empirical_Baseline_BatchB1 | 0.7261 | 44.00 | 26.67 | 29.73 | 0.5100 | 0.2075 | -34.20 |
| Beijing (Info-Rich) | qwen3.7-flash | 2_Catalog_MMR_PostProcessing_Sim (lam_geo=0.25, lam_tail=0.20) | 0.6434 | 54.00 | 10.00 | 36.88 | 0.6320 | 0.2103 | -22.66 |
| Quanzhou (Info-Scarce) | gpt-5.6-sol | 1_Empirical_Baseline_BatchB1 | 0.7514 | 30.00 | 50.00 | 29.39 | 0.5040 | 0.2972 | -67.70 |
| Quanzhou (Info-Scarce) | gpt-5.6-sol | 2_Catalog_MMR_PostProcessing_Sim (lam_geo=0.25, lam_tail=0.20) | 0.6336 | 52.00 | 13.33 | 43.89 | 0.7520 | 0.2855 | -17.56 |
| Quanzhou (Info-Scarce) | claude-sonnet-4-6 | 1_Empirical_Baseline_BatchB1 | 0.7600 | 30.00 | 50.00 | 21.27 | 0.3650 | 0.1229 | -68.13 |
| Quanzhou (Info-Scarce) | claude-sonnet-4-6 | 2_Catalog_MMR_PostProcessing_Sim (lam_geo=0.25, lam_tail=0.20) | 0.6982 | 50.00 | 16.67 | 31.74 | 0.5440 | 0.1245 | -28.08 |
| Quanzhou (Info-Scarce) | deepseek-v4-pro-0813 | 1_Empirical_Baseline_BatchB1 | 0.7576 | 27.00 | 55.00 | 27.58 | 0.4730 | 0.2254 | -74.71 |
| Quanzhou (Info-Scarce) | deepseek-v4-pro-0813 | 2_Catalog_MMR_PostProcessing_Sim (lam_geo=0.25, lam_tail=0.20) | 0.6887 | 50.00 | 16.67 | 38.19 | 0.6550 | 0.2125 | -28.91 |
| Quanzhou (Info-Scarce) | qwen3.7-flash | 1_Empirical_Baseline_BatchB1 | 0.7524 | 32.00 | 46.67 | 30.53 | 0.5230 | 0.1236 | -69.67 |
| Quanzhou (Info-Scarce) | qwen3.7-flash | 2_Catalog_MMR_PostProcessing_Sim (lam_geo=0.25, lam_tail=0.20) | 0.6067 | 55.00 | 8.3300 | 46.32 | 0.7940 | 0.1155 | -12.50 |

#### S8B. Spatial weight sensitivity frontier ($\lambda_2 \\in \\{0.00, 0.15, 0.25, 0.40\\}, \\lambda_3 = 0.20$)

| City | Model | Lambda_Relevance | Lambda_Spatial_MMR | Lambda_LongTail | Gini_OSM | N0_unique_pois | Zero_visibility_pct | Tail_Share_pct | ROSR_tail | NPI_RBO | SDE_Change_vs_OSM_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Beijing (Info-Rich) | gpt-5.6-sol | 0.8000 | 0.0000 | 0.2000 | 0.7105 | 53.00 | 11.67 | 34.31 | 0.5880 | 0.2636 | -38.32 |
| Beijing (Info-Rich) | gpt-5.6-sol | 0.6500 | 0.1500 | 0.2000 | 0.6781 | 55.00 | 8.3300 | 36.88 | 0.6320 | 0.2663 | -30.89 |
| Beijing (Info-Rich) | gpt-5.6-sol | 0.5500 | 0.2500 | 0.2000 | 0.6669 | 54.00 | 10.00 | 38.61 | 0.6620 | 0.2675 | -23.38 |
| Beijing (Info-Rich) | gpt-5.6-sol | 0.4000 | 0.4000 | 0.2000 | 0.6238 | 53.00 | 11.67 | 42.92 | 0.7360 | 0.2834 | -7.7100 |
| Beijing (Info-Rich) | claude-sonnet-4-6 | 0.8000 | 0.0000 | 0.2000 | 0.6781 | 52.00 | 13.33 | 34.31 | 0.5880 | 0.1339 | -36.60 |
| Beijing (Info-Rich) | claude-sonnet-4-6 | 0.6500 | 0.1500 | 0.2000 | 0.6600 | 53.00 | 11.67 | 35.76 | 0.6130 | 0.1358 | -28.47 |
| Beijing (Info-Rich) | claude-sonnet-4-6 | 0.5500 | 0.2500 | 0.2000 | 0.6370 | 53.00 | 11.67 | 38.19 | 0.6550 | 0.1354 | -21.64 |
| Beijing (Info-Rich) | claude-sonnet-4-6 | 0.4000 | 0.4000 | 0.2000 | 0.5903 | 55.00 | 8.3300 | 43.96 | 0.7540 | 0.1387 | -7.8600 |
| Beijing (Info-Rich) | deepseek-v4-pro-0813 | 0.8000 | 0.0000 | 0.2000 | 0.6950 | 52.00 | 13.33 | 31.04 | 0.5320 | 0.1791 | -38.60 |
| Beijing (Info-Rich) | deepseek-v4-pro-0813 | 0.6500 | 0.1500 | 0.2000 | 0.6679 | 54.00 | 10.00 | 33.82 | 0.5800 | 0.1797 | -31.20 |
| Beijing (Info-Rich) | deepseek-v4-pro-0813 | 0.5500 | 0.2500 | 0.2000 | 0.6440 | 54.00 | 10.00 | 36.46 | 0.6250 | 0.1789 | -22.94 |
| Beijing (Info-Rich) | deepseek-v4-pro-0813 | 0.4000 | 0.4000 | 0.2000 | 0.6142 | 56.00 | 6.6700 | 40.90 | 0.7010 | 0.1849 | -9.9400 |
| Beijing (Info-Rich) | qwen3.7-flash | 0.8000 | 0.0000 | 0.2000 | 0.7076 | 55.00 | 8.3300 | 31.11 | 0.5330 | 0.2080 | -35.67 |
| Beijing (Info-Rich) | qwen3.7-flash | 0.6500 | 0.1500 | 0.2000 | 0.6720 | 56.00 | 6.6700 | 34.17 | 0.5860 | 0.2105 | -29.38 |
| Beijing (Info-Rich) | qwen3.7-flash | 0.5500 | 0.2500 | 0.2000 | 0.6434 | 54.00 | 10.00 | 36.88 | 0.6320 | 0.2103 | -22.66 |
| Beijing (Info-Rich) | qwen3.7-flash | 0.4000 | 0.4000 | 0.2000 | 0.5771 | 56.00 | 6.6700 | 42.57 | 0.7300 | 0.2055 | -8.3400 |
| Quanzhou (Info-Scarce) | gpt-5.6-sol | 0.8000 | 0.0000 | 0.2000 | 0.6785 | 51.00 | 15.00 | 38.82 | 0.6650 | 0.2756 | -53.78 |
| Quanzhou (Info-Scarce) | gpt-5.6-sol | 0.6500 | 0.1500 | 0.2000 | 0.6541 | 50.00 | 16.67 | 40.83 | 0.7000 | 0.2754 | -32.96 |
| Quanzhou (Info-Scarce) | gpt-5.6-sol | 0.5500 | 0.2500 | 0.2000 | 0.6336 | 52.00 | 13.33 | 43.89 | 0.7520 | 0.2855 | -17.56 |
| Quanzhou (Info-Scarce) | gpt-5.6-sol | 0.4000 | 0.4000 | 0.2000 | 0.5900 | 54.00 | 10.00 | 50.49 | 0.8650 | 0.2845 | 6.0400 |
| Quanzhou (Info-Scarce) | claude-sonnet-4-6 | 0.8000 | 0.0000 | 0.2000 | 0.7249 | 52.00 | 13.33 | 26.67 | 0.4570 | 0.1132 | -59.73 |
| Quanzhou (Info-Scarce) | claude-sonnet-4-6 | 0.6500 | 0.1500 | 0.2000 | 0.7102 | 52.00 | 13.33 | 28.89 | 0.4950 | 0.1141 | -41.22 |
| Quanzhou (Info-Scarce) | claude-sonnet-4-6 | 0.5500 | 0.2500 | 0.2000 | 0.6982 | 50.00 | 16.67 | 31.74 | 0.5440 | 0.1245 | -28.08 |
| Quanzhou (Info-Scarce) | claude-sonnet-4-6 | 0.4000 | 0.4000 | 0.2000 | 0.6373 | 53.00 | 11.67 | 39.86 | 0.6830 | 0.1339 | -0.2900 |
| Quanzhou (Info-Scarce) | deepseek-v4-pro-0813 | 0.8000 | 0.0000 | 0.2000 | 0.7116 | 52.00 | 13.33 | 33.12 | 0.5680 | 0.2096 | -61.37 |
| Quanzhou (Info-Scarce) | deepseek-v4-pro-0813 | 0.6500 | 0.1500 | 0.2000 | 0.7017 | 50.00 | 16.67 | 35.35 | 0.6060 | 0.2120 | -42.61 |
| Quanzhou (Info-Scarce) | deepseek-v4-pro-0813 | 0.5500 | 0.2500 | 0.2000 | 0.6887 | 50.00 | 16.67 | 38.19 | 0.6550 | 0.2125 | -28.91 |
| Quanzhou (Info-Scarce) | deepseek-v4-pro-0813 | 0.4000 | 0.4000 | 0.2000 | 0.6384 | 53.00 | 11.67 | 44.79 | 0.7680 | 0.2286 | -1.8700 |
| Quanzhou (Info-Scarce) | qwen3.7-flash | 0.8000 | 0.0000 | 0.2000 | 0.6736 | 54.00 | 10.00 | 41.11 | 0.7050 | 0.1024 | -53.48 |
| Quanzhou (Info-Scarce) | qwen3.7-flash | 0.6500 | 0.1500 | 0.2000 | 0.6338 | 54.00 | 10.00 | 44.03 | 0.7550 | 0.1077 | -29.36 |
| Quanzhou (Info-Scarce) | qwen3.7-flash | 0.5500 | 0.2500 | 0.2000 | 0.6067 | 55.00 | 8.3300 | 46.32 | 0.7940 | 0.1155 | -12.50 |
| Quanzhou (Info-Scarce) | qwen3.7-flash | 0.4000 | 0.4000 | 0.2000 | 0.5691 | 57.00 | 5.0000 | 50.83 | 0.8710 | 0.1295 | 10.66 |
