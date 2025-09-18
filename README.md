# AIH_SLI_Predictor  
A method for predicting severe liver fibrosis in AIH


# Requirements   
The program needs to rely on the following versions of Python environment and toolkit to run：    
python >=3.7    
pandas=1.0.1    
scikit-learn=0.22.1      
matplotlib=3.1.3        

# Usage Examples  
## Input：
The input of the model is a sample storage txt file (inpout_X.txt) containing prediction features. Each data in the file is 1X3. The first to fourth columns represent the sample's cell-platelet count, prothrombin international normalized ratio, blood-γ-glutamyl transpeptidase, and gender (0 for female, 1 for male).
demo example as：
    
  ```
    154.0	1.04	128.0
    129.0	1.04	140.0
    78.0	1.33	30.0
  ```

## Output: 
The output of the model is a txt file containing the prediction results. Each line represents the prediction result of a sample and the confidence. 0 represents low liver fibrosis activity and 1 represents high liver fibrosis activity.
demo example as：
  ```
    probability	predicted_label
    0.661742	0
    0.597835	0
    0.601999	1
  ```
When you input the true label of the data and use evaluator to make predictions, the confusion matrix and ROC curve of the model will be output together.

## Usage：  
Run the following command to use the predictor to predict your own data:

  ```
  $ python predictor --model_path ./weights/updated_RF_model_1.pkl --input_X input_data_dir --output_dir output_dir 
  ```

Run the following command to use the evaluator to predict your own data and evalue the performance of the model if you have the true label of your own data:

  ```
  $ python evaluator --model_path ./weights/updated_RF_model_1.pkl --input_X input_X_dir --input_Y input_Y_dir --output_dir output_dir
  ```

