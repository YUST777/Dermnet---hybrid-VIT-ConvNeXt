# ============================================================
# Team Member: Yousef (Project Lead)
# Component: Core AI Architecture (Hybrid ViT + ConvNeXt)
# ============================================================
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
    except RuntimeError as e:
        print(e)
        
# FORCE DISABLE NVIDIA TENSOR CORES (TF32) TO BYPASS CUDNN SHAPE MISMATCHES
tf.config.experimental.enable_tensor_float_32_execution(False)
        
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Concatenate, Input, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
from transformers import TFAutoModel

# 1. CONFIGURATION
IMG_SIZE = 224
BATCH_SIZE = 8  # Reduced to 8 to absolutely guarantee the two massive models fit inside 24GB VRAM
SEED = 42
DATA_DIR = '../data'  # Pointing to the FULL 23-class Kaggle dataset

# 2. DATA GENERATORS (HEAVY AUGMENTATION)
def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,          # ViT and ConvNeXt expect 0-1 or standard normalization
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=SEED
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, test_gen

# 3. HYBRID TRANSFORMER ARCHITECTURE
def build_hybrid_transformer(num_classes):
    print("\n--- Building Vision Transformer (ViT) + ConvNeXt Hybrid ---")
    
    # Shared Input Layer
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # === BRAIN 1: ConvNeXtBase (Local Texture Features) ===
    # SOTA 2024 Convolutional network that mimics Transformer logic natively
    conv_base = ConvNeXtBase(weights='imagenet', include_top=False, input_tensor=inputs)
    conv_base._name = 'convnext_base'
    for layer in conv_base.layers: layer.trainable = False
    x1 = GlobalAveragePooling2D()(conv_base.output)
    
    # === BRAIN 2: Vision Transformer (Global Context Features) ===
    # Using vit-keras Native TensorFlow implementation to bypass HF fragmentation!
    from vit_keras import vit
    vit_base = vit.vit_b16(
        image_size=IMG_SIZE,
        activation='linear',
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )
    vit_base._name = 'vit_base'
    for layer in vit_base.layers: layer.trainable = False
    
    # Native Keras layers map seamlessly without CuDNN transpose faults!
    vit_output = vit_base(inputs)
    x2 = BatchNormalization()(vit_output)

    # === COMBINE BRAINS ===
    merged = Concatenate()([x1, x2])
    
    # SOTA Medical Classification Head
    x = BatchNormalization()(merged)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adamax(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model, conv_base, vit_base

# 4. TRAINING EXECUTION
if __name__ == '__main__':
    print("\n=== INITIALIZING 2026 HYBRID TRANSFORMER TRAINING ===")
    train_gen, test_gen = get_data_generators()
    num_classes = len(train_gen.class_indices)
    
    # We detach the architecture into two distinct frozen brains
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    conv_base = ConvNeXtBase(weights='imagenet', include_top=False, input_tensor=inputs)
    conv_features = GlobalAveragePooling2D()(conv_base.output)
    model_conv = Model(inputs, conv_features)
    
    from vit_keras import vit
    vit_base = vit.vit_b16(image_size=IMG_SIZE, activation='linear', pretrained=True, include_top=False, pretrained_top=False)
    vit_features = vit_base(inputs)
    model_vit = Model(inputs, vit_features)
    
    print("\n=== PHASE 1: EXACT FEATURE EXTRACTION (Bypassing CuDNN Backprop Bug) ===")
    import math
    def extract_features(generator, samples):
        all_conv = []
        all_vit = []
        all_labels = []
        batches = math.ceil(samples / BATCH_SIZE)
        
        for i in range(batches):
            x_batch, y_batch = next(generator)
            c_feat = model_conv.predict(x_batch, verbose=0)
            v_feat = model_vit.predict(x_batch, verbose=0)
            all_conv.append(c_feat)
            all_vit.append(v_feat)
            all_labels.append(y_batch)
            if i % 50 == 0:
                print(f"Extracted {i}/{batches} batches...")
                
        return np.concatenate(all_conv), np.concatenate(all_vit), np.concatenate(all_labels)

    print("Extracting Train Features...")
    train_conv, train_vit, train_labels = extract_features(train_gen, train_gen.samples)
    print("Extracting Test Features...")
    test_conv, test_vit, test_labels = extract_features(test_gen, test_gen.samples)
    
    # Merge Features natively on CPU!
    X_train = np.concatenate([train_conv, train_vit], axis=1)
    X_test = np.concatenate([test_conv, test_vit], axis=1)
    
    # Compute sever class weights
    classes = np.argmax(train_labels, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(classes), y=classes)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    print("\n=== PHASE 2: TRAINING HYBRID HEAD ===")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import CSVLogger
    
    # Build Top Classifier
    top_model = Sequential([
        BatchNormalization(input_shape=(X_train.shape[1],)),
        Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    top_model.compile(optimizer=Adamax(learning_rate=0.001), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
                      
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        ModelCheckpoint('../models/final_hybrid_transformer.keras', save_best_only=True, monitor='val_accuracy', mode='max'),
        CSVLogger('../training_history.csv', append=True)
    ]
    
    top_model.fit(X_train, train_labels,
                  validation_data=(X_test, test_labels),
                  epochs=30, batch_size=32,
                  callbacks=callbacks,
                  class_weight=class_weight_dict)
    
    print("\nBleeding Edge Training Complete! Head saved as final_hybrid_transformer.keras")
