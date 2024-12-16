# used to calculate narrow sense heritability

for trait in $(ls data/plink_traits); do
    ldak --reml Main/results/narrow_sense_h/$trait --grm data/bimbam/k2029.indpw --pheno data/plink_traits/$trait --constrain YES
done

for trait in $(ls data/plink_traits); do
    ldak --reml Main/results/narrow_sense_h/${trait}1 --grm data/bimbam/k2029.indpw2 --pheno data/plink_traits/$trait --constrain YES
done