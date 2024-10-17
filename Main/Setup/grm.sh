plink --bfile k2029 --maf 0.05 --indep-pairwise 200 kb 1 0.6 --out k2029.maf005.indpw
plink --bfile k2029 --extract k2029.maf005.indpw.prune.in --out k2029.indpw
ldak --calc-kins-direct k2029 .indpw --bfile k2029.indpw --ignore-weights YES --power -1



plink --bfile k2029 --maf 0.05 --indep-pairwise 500 kb 1 0.5 --out k2029.maf005.indpw2
plink --bfile k2029 --extract k2029.maf005.indpw.prune.in --out k2029.indpw2
ldak --calc-kins-direct k2029.indpw2 --bfile k2029.indpw --ignore-weights YES --power -1 --extract k2029.maf005.indpw2.prune.in