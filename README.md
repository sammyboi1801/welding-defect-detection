# Welding Defect Prediction(IIT Techfest Weldright 2022!)
2nd Prize solution to IIT Techfest 2022's Weldright Hackathon(dataset provided by Godrej Aerospace)

![image](https://user-images.githubusercontent.com/80597420/208255391-3b45e1cb-7157-4d45-9b83-92818156a169.png)



<b>Problem Statement</b>: Participants were expected to use ML models to predict welding defects in the materials by
developing algorithms using the provided parameters. Participants were free to use any
technique provided it is suited for the variety and volume of data provided.
The end goal of this competition is to enable the Godrej Aerospace team to produce
defect-free products every time.

<b>Dataset Provided</b>: Godrej Aerospace has identified multiple parameters affecting the welding process and recorded the
dataset from past activities. Process parameters such as ambient temperature, weld job
temperature, humidity, voltage current, welding travel speed, shielding gas flow, and metal
composition. For advanced analytics machine data, welder details are provided.

<br>
<h2>Analysis of the dataset</h2>

1) Correlation between the parameters:

    We made a heatmap of Pearson’s correlation of all the different independent variables. We found the correlation between Current and Voltage and also between Temperature and Humidity to be significantly higher than the other parameters(r-value). The correlation between current and voltage is very obvious due to Ohm’s law(V=IR). The positive correlation between ambient temperature and humidity is also a physical phenomena which can be confirmed from our dataset. Hence, no problem of multicollinearity. 

    ![image](https://user-images.githubusercontent.com/80597420/208256030-4b0a9f4a-aa70-43e5-b91e-a925f182d28b.png)



2) Inference regarding Job Temperature:

      Tungsten Inclusion generally takes place when the welding job temperature is really high which causes the tungsten to melt and may enter the weld metal leading to defective welding. This physical property of welding is apparent in the below representation. We can see that Tungsten Inclusion has a noticeably higher ‘Job Temp’. According to this, we can suggest that a temperature range between 30-60 degree celsius would be ideal for welding. Highest defect count on the given data(29.08.22) has relatively high mean job temperature.
      
      ![image](https://user-images.githubusercontent.com/80597420/208256317-4894accb-76a9-4fbc-a94a-dbb409b3b4cc.png)
      ![image](https://user-images.githubusercontent.com/80597420/208256332-9ddda84a-f8a7-46bf-bb07-f18785283f02.png)
      
      
      
3) Inference regarding Current and Voltage:

    From the below table, we can also infer that Tungsten Inclusion has a higher mean Current compared to No defect. This, according to us, is because heat generated is directly proportional to the square of Current. Higher the current, higher the heat produced(high job temperature) and hence, higher chances of Tungsten Inclusion. Similarly, we also observe that average voltage is higher compared to no defect. This suggests that voltage might be affecting Tungsten Inclusion.
    
    <b>H = I2 RT</b>
    <br>
    Where, 
    I = Current , 
    R=Resistance , 
    T=Time and H=Heat energy

    
    ![image](https://user-images.githubusercontent.com/80597420/208256461-fe49b5e1-d463-46f7-9c5b-b3e15ab8a46c.png)



4) The average temperature, humidity and flow rate is similar between No defect and Porosity throughout order operation numbers. This suggests that temperature and humidity have minimal role in porosity. Ideally, high humidity is a major factor for porosity. But, in our dataset, we see that the range of humidity is very limited.

    ![image](https://user-images.githubusercontent.com/80597420/208256509-ec275192-7365-4fcb-8203-413c75bd57c4.png)
    ![image](https://user-images.githubusercontent.com/80597420/208256513-bd203d62-c6ad-4eb8-b242-c2c77ae08b93.png)


5) The average flow rate for porosity is higher than no defect and tungsten inclusion across the entire dataset.

    ![image](https://user-images.githubusercontent.com/80597420/208256558-12170ce7-1208-46bc-ac87-a1e3d0e4c411.png)
    ![image](https://user-images.githubusercontent.com/80597420/208256552-91abe067-8011-4291-988a-de5b97b86251.png)


<h2>Imbalanced data</h2>

We also see the dataset is unbalanced in its representation. We can use many different methods to tackle this. This is really important for training the ML model as we don’t want the model to output biased results. There are three main methods of doing this:

![image](https://user-images.githubusercontent.com/80597420/208257182-e831d0aa-1d5c-45e5-8104-30d2d2d65333.png)

1) <b>Over-sampling</b>: In this method, we basically over-sample the minority categorical dependent variable. We duplicate those values so that we get a larger representation of our minority category. This increases our dataset size which is the last thing we would want to do.
2) <b>Under-sampling</b>: In this method, we take a part of a dataset from the majority category so that we get an equal representation of all the categories.  This method is effective and can be used by ensemble techniques(Random Forest, XGBoost, etc).
3) <b>SMOTE(Synthetic Minority Oversampling Technique)</b>: This creates synthetic data of the minority category to populate it. It uses the KNN algorithm to create meaningful duplication of the minority class. This increases the size too. 

We came with a hybrid solution which uses a combination of all these three methods to create a robust dataset. 

Firstly, we reduced the majority class "No Defect" to about 200,000 (Under-sampling) by random selection and duplicate the minority classes "Porosity" and "Tungsten Inclusion" to get a satisfactory representation of all the classes(Over-sampling).

![image](https://user-images.githubusercontent.com/80597420/208257381-8e5a58b4-5e2b-486a-83e8-435f91dd132d.png)

Once we are done with these two methods, now we can use the SMOTE algorithm to over-sample the minority classes.

![image](https://user-images.githubusercontent.com/80597420/208257449-bd16b494-4ad2-4a7a-a1a8-22a7f517ae85.png)


This leaves us with a robust dataset which can be sent for training!


<h2>Models Used</h2>

![image](https://user-images.githubusercontent.com/80597420/208258287-843ae410-82c0-4ed9-ac1c-8817ebdf5925.png)

![image](https://user-images.githubusercontent.com/80597420/208258511-d2c08702-a70a-4690-8ace-eaf57dfd9efb.png)

![image](https://user-images.githubusercontent.com/80597420/208258569-22757cfb-8b77-406c-bd92-0c970cf503e4.png)

![image](https://user-images.githubusercontent.com/80597420/208258631-d3df1341-3f0e-4d52-9e87-4e507aa3eca4.png)

![image](https://user-images.githubusercontent.com/80597420/208258699-d5a38c9a-9eea-4068-8fec-829cfe957702.png)


1) We observe that SVM and Adaboost are not giving a good accuracy score. This suggests that model is clearly under-fitting on the data. Hence, we had to reject both these models.

2) We also see that Decision Tree and Random Forest is giving an accuracy of 100%. This is clearly overfitting. Using these models will reflect a very poor testing accuracy score.

3) In the end, we tried out the normal gradient boosting algorithm. This model over-the-top gave us a good accuracy score. We decided this could be a good starting point to reach to our final deployable model.

Finally, we decided that XGBoost, an advanced version of gradient boosting algorithm, can be used to train our model.

![image](https://user-images.githubusercontent.com/80597420/208260799-76798c51-8044-45b6-970f-08ba3881cf81.png)

As we can see, XGBoost gave us a really good accuracy(96%) and F1-score. This was achieved after tuning parameters such as n_estimators.


<h2>Deploying the model</h2>
We created a simple web page using Tailwind CSS and used a Flask server for deploying it on AWS EC2.

Our model is really efficient and easily runs on the resources available on the Free Tier provided by AWS. Our model hardly takes 30% of 1GB RAM while running. The total size of the model is just 35MB. This makes this model really suitable for scaling. 

The cost of operation of this model is minimal, which can be our calculation of Total Cost of Ownership(TCO) of our model!


<hr>


NOTE:

'xgboost-350' : This is the zipped file which has our xgboost model(.pkl)

'main.py' : This has the code for the flask server that was used for deploying our model.

'227088_WeldRight_Abstract' : This has data analysis of the dataset that we did.

'Weldright.ipynb' : This has the final code for using XGBoost algorithm

'templates' : This directory has the html file for the UI we made.


<hr>


You can find all the files for our project over here. 

Happy Learning :)











    










