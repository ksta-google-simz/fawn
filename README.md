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
### Num Inference
<table>
  <tr>
    <td>
      <p><img src="https://github.com/user-attachments/assets/c0d03587-4907-421c-8809-b3b317aaa111" alt="image"/></p>
    </td>
    <td>
      <p>We conducted comparative experiments using num_inf values of 10 and 25 at three anonymization degrees: 0.75, 1.00, and 1.25. (Fig 1)</p>
        <p>The average inference times were measured as follows:</p>
        <ui>
            <li>10 steps: approximately 8.716 seconds per image</li>
            <li>25 steps: approximately 18.609 seconds per image</li>
        </ui>
        <p></p>
        <p>The results revealed no significant performance gains when increasing num_inf to 25. In some cases, using 10 steps even produced better outcomes. Based on these observations, we decided to set num_inf=10 for our experiments.</p>
    </td>
  </tr>
</table>

### Anonymization Degree
<table>
  <tr>
    <td>
      <p><img src="https://github.com/user-attachments/assets/6c94087c-0a4d-47ff-acf3-4b3ddec62d1d" alt="image"/></p>
    </td>
    <td>
      <p>We conducted comparative experiments using num_inf values of 10 and 25 at three anonymization degrees: 0.75, 1.00, and 1.25. (Fig 1)</p>
        <p>After normalizing the Re-ID and Avg Similarity metrics, we integrated them with the AGR metric by assigning weights of 0.25, 0.25, and 0.5 respectively :
$$Score_i = w_1 R'_i + w_2 S'_i + w_3 \, AGR_i$$</p>
        <p>This composite score was used to determine the optimal anonymization degree.</p>
        <p>Based on our experiments, the highest score (0.537) was achieved at an anonymization degree of 1.02, which we selected as the optimal setting. (Fig 4)</p>
    </td>
  </tr>
</table>

### Comparative Experiments: Paper’s Proposed Model vs. Our Tuned Model
</br>
<table>
  <tr>
    <td>
        <p><img src="https://github.com/user-attachments/assets/6cba765e-440a-4916-9977-02e4a0e4c0ce" alt="image" </p>
        <p><img src="https://github.com/user-attachments/assets/aacb29d4-bb5e-4202-b3bd-c5a05397e64e" alt="image" </p>
    </td>
    <td>
      <p>As shown in Table 2, our tuned model demonstrates a noticeable decrease in Re-ID while achieving improvements in FID and AGR compared to the original paper’s model on the CelebA-HQ dataset.</p>
        <p>Notably, the naturalness of the anonymized images is better preserved, and gender accuracy (Gender Acc.) increases significantly from about 45% to nearly 80%.</p>
        <p>These results indicate that our approach not only lowers re-identification risk but also enhances overall image quality and attribute recognition performance.</p>
    </td>
  </tr>
</table>


## 🗒️ References
[1] H.-W. Kung, T. Varanka, S. Saha, T. Sim, and N. Sebe,
“Face Anonymization Made Simple,” arXiv preprint arXiv:2411.00762, 2024.

[2] A. Serengil, “DeepFace: A Python library for face recognition and facial attribute analysis,” GitHub repository, https://github.com/serengil/deepface, accessed Mar. 26, 2025.

[3] K. Karkkainen and J. Joo, “FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021.

[4] T. Karras, T. Aila, S. Laine, and J. Lehtinen, “CelebA-HQ: High-Quality Images for Face Attributes,” arXiv preprint arXiv:1710.10196, 2017.

[5] T. Karras, S. Laine, and T. Aila, “FFHQ: A High-Quality Dataset of Human Faces,” arXiv preprint arXiv:1907.09661, 2019.

---
🛠️ **project for** : KSTA NIPA Google ML Bootcamp Project 


