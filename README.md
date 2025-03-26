# F.A.W.N
: Face Anonymization With Neural-networks

An selective face anonymization AI system that anonymizes all faces except for pre-registered individuals. This project extends the model [`face_anon_simple`](https://github.com/hanweikung/face_anon_simple) by adding
- ☑️ **Identity exclusion logic** for registered faces
- 📊 **Performace evaluation metircs** including FID, ReID, and AGR(AGR model: [`serengil_deepface`](https://github.com/serengil/deepface))
- 🌐 **Serverless web interface** built with Gradio

---
## 🔍 Project Goal
- Develop a **selective face anonymization system** that excludes user-specified individuals from anonymization.
- Enhance the anonymization quality and inference speed of the based model by **tuning hyperparameters**.
- Evaluate the anonymization quality of the tuned model using **multiple objective and perceptual metrics**.

## 🧪 Key Experiments
We conducted extensive experiments by adjusting the following hyperparmeters:
- `num_inference_steps` : number of diffusion steps (-> image quality and speed)
- `anonymization_degree` : degree of face transformation


## 📊 Evaluation Metrics
To evaluate the anonymization results from newly tuned model, we used:
- **Re-ID Rate & Avg Similarity**: Both metrics aim to assess the similarity between anonymized images and their corresponding original images using cosine similarity. However, they differ in their computation:
    - **Re-ID Rate**: Computed by comparing each anonymized image against the entire dataset. An image is considered correctly re-identified if the highest similarity score corresponds to its original version, and the rate is obtained by dividing the correct count by the total number of images.
    - **Avg Similarity**: Calculated as the average cosine similarity between each original image and its anonymized counterpart across the dataset, providing a continuous measure of overall similarity.
- **FID(Frechet inception Distance)** : Assess the visual quality of anonymized images through the PyTorch FID module.
- **AGR Similarity**: Quantifies how naturally anonymized faces preserve Age, Gender, and Race attributes by comparing them between the original and anonymized images. This metric is critical in our project because it ensures that while faces are effectively anonymized, essential demographic features remain natural and recognizable. AGR Similarity bridges the gap between robust privacy protection and maintaining the intrinsic quality of facial attributes. The AGR indicators are calculated as follows : 
$$AGR = \alpha \cdot Age Dist. + \beta \cdot Gender Acc + \gamma \cdot Race Acc$$

## 💭 Results & Findings
<img src="https://github.com/user-attachments/assets/c0d03587-4907-421c-8809-b3b317aaa111" alt="image" height="180"/>
<img src="https://github.com/user-attachments/assets/df8deba6-fda6-44c6-886f-30bc67a1ca81" alt="image" height="180"/>

### Num Inference
- We conducted comparative experiments using num_inf values of 10 and 25 at three anonymization degrees: 0.75, 1.00, and 1.25. (Fig 1)
- The average inference times were measured as follows:
    - 10 steps: approximately 8.716 seconds per image
    - 25 steps: approximately 18.609 seconds per image
- The results revealed no significant performance gains when increasing num_inf to 25. In some cases, using 10 steps even produced better outcomes. Based on these observations, we decided to set num_inf=10 for our experiments.
### Anonymization Degree
- After normalizing the Re-ID and Avg Similarity metrics, we integrated them with the AGR metric by assigning weights of 0.25, 0.25, and 0.5 respectively :
$$Score_i = w_1 R'_i + w_2 S'_i + w_3 \, AGR_i$$
- This composite score was used to determine the optimal anonymization degree.
- Based on our experiments, the highest score (0.537) was achieved at an anonymization degree of 1.02, which we selected as the optimal setting. (Fig 4)

## 🗒️ Sample Result Table


## 🧠 Insights

---
🛠️ **project for** : KSTA NIPA Google ML Bootcamp Project 


