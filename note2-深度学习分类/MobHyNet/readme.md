MobHyNet Model Architecture


MobHyNet is a multi-modal deep learning architecture designed to integrate multiple imaging modalities, such as different types or views of ultrasound images, with structured clinical features for classification tasks (for example, disease diagnosis or subtyping). The model begins by using separate MobileNetV2 backbones to extract deep features from each imaging modality. Each MobileNetV2 acts as an independent encoder, generating high-level representations of their respective inputs without the original classifier head. Clinical (structured) data is processed in parallel through a small fully connected network, which transforms and projects these features into a 128-dimensional space.

After feature extraction, all modality-specific image features are concatenated and combined with the processed clinical features to form a unified feature vector. This combined representation is then fused and reduced in dimension by a fully connected layer to 512 dimensions. To further enhance the model’s ability to capture cross-modal and global dependencies, a multi-head self-attention mechanism is applied to the fused features. This attention layer enables the model to dynamically weigh the importance and interactions among components of the multi-modal representation.

Finally, the attended features are passed through a normalization layer and a series of fully connected layers with non-linear activation, culminating in the output layer that produces class predictions (in this example, a four-class classification). The overall design emphasizes efficient feature extraction (via MobileNetV2), effective modality information integration, and enhanced representation learning using attention mechanisms, making MobHyNet well-suited for clinical prediction tasks involving both imaging and non-imaging data.



MobHyNet training and testing was performed using Pixelmed AI.
