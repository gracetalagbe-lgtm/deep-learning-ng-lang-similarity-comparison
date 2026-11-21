---
language:
- afr
- amh
- arb
- bem
- ewe
- fra
- hau
- ibo
- kin
- mey
- mlg
- msa
- nde
- nso
- nya
- orm
- shi
- sna
- som
- ssw
- swa
- tam
- tel
- tir
- ton
- tsn
- urd
- ven
- wol
- xho
- yor
- zul
pretty_name: ntrex_african
license:
  - cc-by-sa-4.0
multilinguality:
  - translation
task_categories:
  - translation
source_datasets:
- ntrex
size_categories:
  - '1997'
configs:
- config_name: afr_Latn
  data_files:
  - split: test
    path: ntrex_african/afr_Latn.tsv
- config_name: amh_Ethi
  data_files:
  - split: test
    path: ntrex_african/amh_Ethi.tsv
- config_name: arb_Arab
  data_files:
  - split: test
    path: ntrex_african/arb_Arab.tsv
- config_name: bem_Latn
  data_files:
  - split: test
    path: ntrex_african/bem_Latn.tsv
- config_name: ewe_Latn
  data_files:
  - split: test
    path: ntrex_african/ewe_Latn.tsv
- config_name: fra_Latn
  data_files:
  - split: test
    path: ntrex_african/fra_Latn.tsv
- config_name: hau_Latn
  data_files:
  - split: test
    path: ntrex_african/hau_Latn.tsv
- config_name: ibo_Latn
  data_files:
  - split: test
    path: ntrex_african/ibo_Latn.tsv
- config_name: kin_Latn
  data_files:
  - split: test
    path: ntrex_african/kin_Latn.tsv
- config_name: mey_Arab
  data_files:
  - split: test
    path: ntrex_african/mey_Arab.tsv
- config_name: mlg_Latn
  data_files:
  - split: test
    path: ntrex_african/mlg_Latn.tsv
- config_name: msa_Latn
  data_files:
  - split: test
    path: ntrex_african/msa_Latn.tsv
- config_name: nde_Latn
  data_files:
  - split: test
    path: ntrex_african/nde_Latn.tsv
- config_name: nso_Latn
  data_files:
  - split: test
    path: ntrex_african/nso_Latn.tsv
- config_name: nya_Latn
  data_files:
  - split: test
    path: ntrex_african/nya_Latn.tsv
- config_name: orm_Ethi
  data_files:
  - split: test
    path: ntrex_african/orm_Ethi.tsv
- config_name: shi_Arab
  data_files:
  - split: test
    path: ntrex_african/shi_Arab.tsv
- config_name: sna_Latn
  data_files:
  - split: test
    path: ntrex_african/sna_Latn.tsv
- config_name: som_Latn
  data_files:
  - split: test
    path: ntrex_african/som_Latn.tsv
- config_name: ssw_Latn
  data_files:
  - split: test
    path: ntrex_african/ssw_Latn.tsv
- config_name: swa_Latn
  data_files:
  - split: test
    path: ntrex_african/swa_Latn.tsv
- config_name: tam_Taml
  data_files:
  - split: test
    path: ntrex_african/tam_Taml.tsv
- config_name: tel_Telu
  data_files:
  - split: test
    path: ntrex_african/tel_Telu.tsv
- config_name: tir_Ethi
  data_files:
  - split: test
    path: ntrex_african/tir_Ethi.tsv
- config_name: ton_Latn
  data_files:
  - split: test
    path: ntrex_african/ton_Latn.tsv
- config_name: tsn_Latn
  data_files:
  - split: test
    path: ntrex_african/tsn_Latn.tsv
- config_name: urd_Arab
  data_files:
  - split: test
    path: ntrex_african/urd_Arab.tsv
- config_name: ven_Latn
  data_files:
  - split: test
    path: ntrex_african/ven_Latn.tsv
- config_name: wol_Latn
  data_files:
  - split: test
    path: ntrex_african/wol_Latn.tsv
- config_name: xho_Latn
  data_files:
  - split: test
    path: ntrex_african/xho_Latn.tsv
- config_name: yor_Latn
  data_files:
  - split: test
    path: ntrex_african/yor_Latn.tsv
- config_name: zul_Latn
  data_files:
  - split: test
    path: ntrex_african/zul_Latn.tsv
---

### Dataset Summary

Multilingual News Test References for MT Evaluation from English into 32 target African languages in the NTREX-128 dataset. Adapted from [NTREX]{https://github.com/MicrosoftTranslator/NTREX/tree/main}, processed into tab-separated value (TSV) files for easy integration into evaluation workflows

### Sample usage

```python
from datasets import load_dataset

# Load a specific language configuration
dataset = load_dataset("masakhane/ntrex_african", name="afr_Latn", split="test")
```

### Languages 
- afr_Latn
- amh_Ethi
- arb_Arab
- bem_Latn
- ewe_Latn
- fra_Latn
- hau_Latn
- ibo_Latn
- kin_Latn
- mey_Arab
- mlg_Latn
- msa_Latn
- nde_Latn
- nso_Latn
- nya_Latn
- orm_Ethi
- shi_Arab
- sna_Latn
- som_Latn
- ssw_Latn
- swa_Latn
- tam_Taml
- tel_Telu
- tir_Ethi
- ton_Latn
- tsn_Latn
- urd_Arab
- ven_Latn
- wol_Latn
- xho_Latn
- yor_Latn
- zul_Latn
