The results from different tests will be saved here as CSV files. The folder contains:

1. `filters.csv` - the statistics results about the filters.

   Result format:
   | target   |    attacker    | distance      |
   |----------|----------------|---------------|
   |    T1    |        A1      |     0.9985    |
   
2. `targeted_impersonation_results.csv` - the simulated impersonation results for targeted impersonation.
   
   Result format:
   | target   |    attacker    | laser setting |
   |----------|----------------|---------------|
   |    T1    |        A1      |     100       |
   
3. `untargeted_impersonation_results.csv` - the simulated impersonation results for predictable untargeted impersonation.
   
    Result format:
   | attacker | laser setting  | identities |
   |----------|----------------|------------|
   |    A1    |        90      |     T1     |

4. `celeb_results.csv` - the celebrity recognition results.
   
   Result format:
   | attacker | celebrity name | confidence | Urls    |
   |----------|----------------|------------|---------|
   |photo name|  Gayatri Nair  |    78.89   |'www.wikidata.org/wiki/Q5528725'|
   
5. `id_dodging.csv` - the simulated identity dodging results.

    Result format:
   | attacker | laser setting  | distance |
   |----------|----------------|------------|
   |    A1    |        220      |    1.428     |
   
6. `db_dodging.csv` - the simulated database dodging results.

   Result format:
   | failed  identity | laser setting  |
   |---------- |----------------|
   |    T10    |        200      |    

