# Predicting heavy metal adsorption in soil
A machine learning model based on gradient boosting decision tree for predicting heavy metal adsorption in soil.

An online predictor has been created on the project [**ChemAI**](https://www.chemai.aropha.com/) launched by Dr. Huichun Zhang's research group at Case Western Reserve University, which is hosted and supported by [**Aropha Inc.**](https://www.aropha.com/) at: **https://www.chemai.aropha.com/adsorption/heavy-metal-in-soil/about.html**

![Adsorption_HMsoil_TOC](https://user-images.githubusercontent.com/70991409/138607519-976e7eda-66fe-46c1-b591-0b59c15b8d75.png)


## Dataset
The regression model was built on 4,420 data points for soil adsorption to 6 heavy metals (i.e., Cd, Cr, Cu, Pb, Ni, Zn). The model included 9 inputs: (i) four descriptors for soil properties, namely, pH of soil, CEC (cation exchange capacity, cmol/kg), OC (organic carbon, %), and clay content(%); (ii) five descriptors for the adsorption system, namely, the equilibrium concentration (Ce,mg/L), solution pH, ionic strength (I, mol/L), temperature (T, °C), and soil-to-solution ratio (g/mL). The output was the natural logarithm of the corresponding adsorbed heavy metal amount on soil (Ln-mg/g).

## ML algorithms
A total of 10 ML algorithms were examined to find the best one, including Classification and Regression Trees, K-nearest neighbors, Linear regression, Ridge regression, Stochastic gradient descent regressor, Support vector regression, Extremely randomized trees, Gradient boosting decision tree, Random forest, Extreme gradient boosting.

Gradient boosting decision tree was found to be the best one.

## Other notes
The feature importance of 9 input parameters was analyzed using the Shapley additive explanation method to identify their effect on the adsorption, which agreed with the known mechanisms.

## Publication
This work has been published on Environmental Science & Technology: 

Yang, H.; Huang, K.; Zhang, K.; Weng, Q.; Zhang, H.; Wang, F. Predicting Heavy Metal Adsorption on Soil with Machine Learning and Mapping Global Distribution of Soil Adsorption Capacities. *Environ. Sci. Technol.* **2021**, *55* (20), 14316-14328. https://doi.org/10.1021/acs.est.1c0247

![Adsorption_HMsoil_plot](https://user-images.githubusercontent.com/70991409/138607531-5f74f1ec-fa7d-4c70-8237-f334e85bc464.png)
