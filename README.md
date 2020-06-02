## Data Collection and Processing

The data used are available here ["Cook County District Attorney Site"](https://datacatalog.cookcountyil.gov/browse?tags=state%27s+attorney+case-level&sortBy=most_accessed). These data included case information for felonies that were charged in the district attorney office’s jurisdiction from 2011-2016. Key variables included information regarding the charges (including charge severity), the date and time of the alleged incident, the defendant’s age, race, and gender, and various information about the arresting authority as well as the court authority involved (courthouse, judge, etc.). The dataset lacked criminal history information--a key omission. Please see the report for a discussion of this omission.

For simplicity’s sake, in this example, we decided to focus on just one crime category: retail theft. For the same reason, we focused on only the two most common racial groups listed for defendants: White and Black. 

In this jurisdiction, charges have different classes. X is the most severe charge class, followed by 1 through 4, with 4 being the least severe charge class; and, also, there are misdemeanors, which are less severe than a 4. A charge reduction is when a defendant is initially charged with an offense of a certain degree but receives a plea bargain in which the most severe guilty charge is less severe than what was initially charged. For example, if a defendant is charged with a class 1 felony but receives a class 2 felony via plea bargain, then the defendant has received a charge reduction. We chose charge reductions as our outcome variable since its usefulness to prosecutors is clear and since it is an intuitive variable. Also, charge reductions are a good choice because they represent a clear decisional point.

The code for Data Processing is in `Data Processing & Transformation.Rmd` from lines 1-139.

Note: in the report, we are approach agnostic, recommending neither this nor any specific fairness approach. Rather, in this report, we have outlined how an office might develop an approach that is tailored to its specific context and vetted, iteratively, in order to ensure fairness (see Report section titled, "Creating a Framework for Using AI to Remove Bias in Decision Making"). Indeed, we believe model creation is the core of the process and will require continuous discussion and revisiting.

## Purely Predictive Model

For our purely predictive model, we built a predictive model that reflects how a defendant historically would have been treated. The code for creating the purely predictive model can be found in two places:

The code for creating the purely predictive model and interpreting results is in two places:
* `Predictive & Suggestive Models.Rmd` from lines 8-72
* `Predictive & Suggestive Models_v2.Rmd` from lines 3-82

To replicate the results in the report, use v2.

## Race-Neutral Predictive Model

For our race-neutral model, we used an approach that penalizes unfairness in the model training process. Specifically, we applied a method designed by algorithmic scholars Yahav Bechavod and Katrina Ligett in ["Penalizing Unfairness in Binary Classification"](https://arxiv.org/abs/1707.00044), wherein two group-dependent regularization terms are added to the loss function. These terms penalize differences in the false positive rates (FPR) and false negative rates (FNR) between groups defined by a protected variable—in this instance, race. As these scholars note, this approach can be used with different types of models, including support vector machines (SVM) and, that which we use here, logistic regression. We used logistic regression as our base algorithm because it was fully tested in Bechavod and Ligett’s published work. This model is our modified predictive model: it describes, in essence, how the case would resolve if the defendant is treated race-neutrally.

Predictions with fitted model can be run immediately in `race_agnostic_model.Rmd`. To retrain model follow the instructions below.

Note: We include this model because we believe it serves a cautionary tale. In terms of overall charge reduction rates, the model did not decrease racial disparity by much. Why might this be? Because the historical case record itself is biased, bounding a model so that FPR and FNR are held relatively equal across groups results in a model that is “fair,” in a sense, but it is not a model that addresses the underlying racial disparities embedded in the historical record, such as those reflected in proxies for race, including a defendant’s municipality or the court facility in which a case was processed. Rather, it is a model that, if used without contextual awareness and delicacy, threatens to enshrine such disparities moving forward.

### Training

The code to train this model and find hyperparameters is automated in the following github repository. Dependencies are automatically downloaded following the instructions in their README.

    `$ git clone https://github.com/jjgold012/lab-project-fairness.git`

To train a new model with our Chicago dataset we provide code to create the neccesary inputs to run their method. These include a reformated training data set and a json file which outlines options for their method to use, such as target variable, features, and protected group variable.

    `$ python model3_preprocess.py train.csv test.csv`

Three files will be created, `train_processed.csv`, `test_processed.csv`, and `chicago.json`. Two of the files must be moved to run the fairness method. Model predictions will be made using `test_processed.csv` in the Princeton-CLSJ-PBM-Project directory.

    `$ cp train_processed.csv ~/lab-project-fairness/fairness_project/datasets/`
    `$ cp chicago.json ~/lab-project-fairness/fairness_project/options/`

Run the training proceedure.

    `$ python ~/lab-project-fairness/fairness_project/start.py ~/lab-project-fairness/fairness_project/options/chicago.json > chicago.result`

The method will split the training set via 5-fold cross validation to suggest the best hyperparameters to be used, and returns many outputs which includes the best weight vector used in the trained logistic regression function. Note, test_processed.csv is not being used in the 5-fold cross validation.

### Testing

Fitted weights can be found near the end of the result file under the header "Best Values for Objective squared relaxation" with key "w". Unfortuneately, the authors do not provide an automated way to retrieve or parse results. Copy weights from `chicago.result` to `race_agnost_modelweights.csv`

To predict on `train_processed.csv` run code in `race_agnostic_model.Rmd`.

## Suggestive Model

Third, and most importantly, we created a “suggestive model," showing how a case would resolve if the defendant were treated as if he or she were White. 

Note: When trying to replicate our work with `Predictive & Suggestive Models_v2.Rmd`, we have a prepared file in the Data folder of this repository called `blackandwhite.csv'. We recommend using this file as your data input for this code only. Additionally, we have described the filters applied to the original dataset to create it. 

The code to alter race and covariates of race in order to examine how these may alter the treatment of specific cases is in two places:
* `Data Processing & Transformation.Rmd` from lines 143-260
* `Predictive & Suggestive Models_v2.Rmd` from lines 86-157 (refer to appendix for additional notes)

The code for creating suggestive model predictions and interpreting results is in two places:
* `Predictive & Suggestive Models.Rmd` from lines 76-130
* `Predictive & Suggestive Models_v2.Rmd` from lines 161-280 (refer to appendix for additional notes)

Users may try either approach; however, to replicate the results in the report, use v2.

