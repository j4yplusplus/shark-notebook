# Shark Tooth Morphology Analysis Using PCA

## Overview
This project analyzes morphological measurements from over 99 fossil shark tooth specimens using principal component analysis. The goal was to determine which measurements contribute most strongly to morphological variation and explore patterns in tooth morphology among several understudied shark taxa.

This analysis was apart of a broader research project on understudied extinct shark taxa. I co-authored the resulting research paper and presented at the Jackson School of Geosciences' 15th Annual Student Research Symposium.

This is a living document. PCA was used as a step in data exploration for further research.

## My Contributions
- Cleaned and prepared morphological measurement data for analysis.
- Handled missing data using overall median imputation and later taxon median imputation.
- Standardized quantitative variables prior to PCA.
- Performed Principal Component Analysis using Python and scikit-learn.
- Created visualizations to compare PCA results by taxa.
- Analyzed which measurements contributed most to each principal component.

## Limitations
Degraded quality of the tooth specimen resulted in many missing measurements, handled by median imputation. Median imputations can reduce variability and affect the resulting PCA. Additionally, later use of median imputation by taxon can make taxon look more distinct in PCA than they really are.
