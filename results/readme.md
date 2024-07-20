The results from different tests will be saved here as CSV files. The folder contains:

1. `targeted_pairs.csv` - the statistics result about the filters on targeted attacks.

   Result format:
   | target   |    attacker    | distance      |
   |----------|----------------|---------------|
   |    T1    |        A1      |     0.9985    |

2. `untargeted_pairs.csv` - the statistics result about the filters on targeted attacks.

   Result format:
   | attacker   |    target    | distance      |
   |----------|----------------|---------------|
   |    A1    |        T1      |     0.9985    |
   
3. `targeted_impersonation_results.csv` - the simulated impersonation results for targeted impersonation.
   
   Result format:
   | target   |    attacker    | laser setting |
   |----------|----------------|---------------|
   |    T1    |        A1      |     100       |
   
4. `untargeted_impersonation_results.csv` - the simulated impersonation results for predictable untargeted impersonation.
   
    Result format:
   | attacker | laser setting  | identities |
   |----------|----------------|------------|
   |    A1    |        90      |     T1     |

5. `celeb_results.csv` - the celebrity recognition results.
   
   Result format:
   | attacker | celebrity name | confidence | Urls    |
   |----------|----------------|------------|---------|
   |photo name|  Gayatri Nair  |    78.89   |'www.wikidata.org/wiki/Q5528725'|
   
6. `id_dodging.csv` - the simulated identity dodging results.

    Result format:
   | attacker | laser setting  | distance |
   |----------|----------------|------------|
   |    A1    |        220      |    1.428     |
   
7. `db_dodging.csv` - the simulated database dodging results.

   Result format:
   | failed  identity | laser setting  |
   |---------- |----------------|
   |    T10    |        200      |    

