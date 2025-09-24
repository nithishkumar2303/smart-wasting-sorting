import os
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics as kmetrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def discover_classes(data_dir):
    """Auto-detect which class folders exist"""
    classes = []
    for folder in ['biodegradable', 'recyclable', 'landfill']:
        path = os.path.join(data_dir, folder)
        if os.path.isdir(path) and len(os.listdir(path)) > 0:
            classes.append(folder)
    return sorted(classes)

class WasteClassificationEfficientNet:
    def __init__(self, input_shape=(224, 224, 3), class_names=None):
        self.input_shape = input_shape
        self.class_names = class_names or ['biodegradable', 'recyclable', 'landfill']
        self.num_classes = len(self.class_names)
        self.model = None
        
    def create_model(self, use_pretrained=True):
        print(f"Creating EfficientNet-B0 model (pretrained={use_pretrained})")
        inputs = keras.Input(shape=(224, 224, 3), name="input_rgb")

        weights = "imagenet" if use_pretrained else None
        self.used_pretrained = use_pretrained
        try:
            base = keras.applications.EfficientNetB0(
                include_top=False, weights=weights, input_tensor=inputs
                )
            print("Base input shape:", base.input_shape)
        except Exception as e:
            print("EfficientNet pretrained load failed, falling back to random init:", e)
            base = keras.applications.EfficientNetB0(
                include_top=False, weights=None, input_tensor=inputs
            )
            self.used_pretrained = False

        # If pretrained failed: train backbone from the start
        base.trainable = not self.used_pretrained

        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        self.model = keras.Model(inputs, outputs)
        self.base_model = base
        return self.model
    
    def compile_model(self, learning_rate=1e-4):
        """Compile with optimizer and metrics"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                     kmetrics.Precision(name='precision'), 
                     kmetrics.Recall(name='recall')]
        )
    
    def create_data_generators(self, data_dir, batch_size=32, validation_split=0.2):
        """Create data generators with EfficientNet preprocessing"""
        # Data augmentation for training
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.efficientnet.preprocess_input,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # No augmentation for validation
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.efficientnet.preprocess_input,
            validation_split=validation_split
        )
        
        train_gen = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            classes=self.class_names,
            shuffle=True
        )
        
        val_gen = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            classes=self.class_names,
            shuffle=False
        )
        
        return train_gen, val_gen
    
    def compute_class_weights(self, train_gen):
        """Compute class weights for imbalanced data"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_gen.classes),
            y=train_gen.classes
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Print class distribution
        unique, counts = np.unique(train_gen.classes, return_counts=True)
        print("\nClass distribution:")
        for i, (cls_idx, count) in enumerate(zip(unique, counts)):
            cls_name = list(train_gen.class_indices.keys())[cls_idx]
            weight = class_weight_dict[cls_idx]
            print(f"  {cls_name}: {count} samples (weight: {weight:.2f})")
        
        return class_weight_dict
    
    def train(self, data_dir, epochs=8, batch_size=32, fine_tune_epochs=4):
        import numpy as np, os
        os.makedirs("model", exist_ok=True)
        train_gen, val_gen = self.create_data_generators(data_dir, batch_size)

        # class weights with cap to avoid instability
        cls_ids, counts = np.unique(train_gen.classes, return_counts=True)
        total = counts.sum()
        raw = {int(i): float(total/(len(counts)*c)) for i, c in zip(cls_ids, counts)}
        class_weight = {i: float(min(w, 25.0)) for i, w in raw.items()}
        print("Class counts:", dict(zip(cls_ids, counts)))
        print("Class weights (capped):", class_weight)

        cbs = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint("model/efficientnet_checkpoint.h5", save_best_only=True, monitor="val_accuracy"),
        ]

        if getattr(self, "used_pretrained", False):
            print("\n=== Phase 1: Training with frozen base (pretrained) ===")
            self.model.compile(optimizer=keras.optimizers.Adam(1e-4),
                           loss="categorical_crossentropy",
                           metrics=["accuracy", kmetrics.Precision(name="precision"), kmetrics.Recall(name="recall")])
            h1 = self.model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=cbs,
                            class_weight=class_weight, verbose=1)

            print("\n=== Phase 2: Fine-tuning entire model ===")
            self.base_model.trainable = True
            self.model.compile(optimizer=keras.optimizers.Adam(1e-5),
                           loss="categorical_crossentropy",
                           metrics=["accuracy", kmetrics.Precision(name="precision"), kmetrics.Recall(name="recall")])
            h2 = self.model.fit(train_gen, epochs=fine_tune_epochs, validation_data=val_gen, callbacks=cbs,
                            class_weight=class_weight, verbose=1)
        else:
            print("\n=== Training from scratch (no pretrained) ===")
            self.model.compile(optimizer=keras.optimizers.Adam(1e-3),
                           loss="categorical_crossentropy",
                           metrics=["accuracy", kmetrics.Precision(name="precision"), kmetrics.Recall(name="recall")])
            h1 = self.model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=cbs,
                            class_weight=class_weight, verbose=1)

            self.model.compile(optimizer=keras.optimizers.Adam(1e-4),
                           loss="categorical_crossentropy",
                           metrics=["accuracy", kmetrics.Precision(name="precision"), kmetrics.Recall(name="recall")])
            h2 = self.model.fit(train_gen, epochs=fine_tune_epochs, validation_data=val_gen, callbacks=cbs,
                            class_weight=class_weight, verbose=1)

        return h1, h2, train_gen, val_gen
    
    def evaluate_model(self, val_gen):
        """Detailed evaluation with per-class metrics"""
        print("\n=== Model Evaluation ===")
        predictions = self.model.predict(val_gen, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_gen.classes
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return report, cm
    
    def export_tflite(self, output_path='model/waste_efficientnet.tflite', quantize=True):
        """Export to TFLite with optional quantization"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"\nTFLite model saved to: {output_path}")
        print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        
        # Save metadata
        model_info = {
            'model_type': 'EfficientNet-B0',
            'class_names': self.class_names,
            'input_shape': self.input_shape,
            'preprocessing': 'efficientnet',
            'created_at': datetime.now().isoformat(),
            'model_size_mb': len(tflite_model) / 1024 / 1024
        }
        
        info_path = output_path.replace('.tflite', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Initial training epochs')
    parser.add_argument('--fine-epochs', type=int, default=8, help='Fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--no-pretrained', action='store_true', help='Train from scratch')
    args = parser.parse_args()
    
    # Resolve data directory
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        data_dir = os.path.join('..', 'data')
    
    print("Smart Waste Classification - EfficientNet-B0")
    print("=" * 50)
    print(f"Data directory: {os.path.abspath(data_dir)}")
    
    # Discover available classes
    class_names = discover_classes(data_dir)
    print(f"Classes found: {class_names}")
    
    if len(class_names) < 2:
        print("ERROR: Need at least 2 classes with images")
        return
    
    # Create and train model
    trainer = WasteClassificationEfficientNet(
        input_shape=(224, 224, 3),
        class_names=class_names
    )
    
    # Create model
    trainer.create_model(use_pretrained=not args.no_pretrained)
    trainer.compile_model()
    
    print(f"\nModel parameters: {trainer.model.count_params():,}")
    
    # Train
    history1, history2, train_gen, val_gen = trainer.train(
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fine_tune_epochs=args.fine_epochs
    )
    
    # Evaluate
    report, cm = trainer.evaluate_model(val_gen)
    
    # Save evaluation report
    with open('model/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Export models
    trainer.export_tflite('model/waste_efficientnet.tflite', quantize=True)
    trainer.model.save('model/waste_efficientnet_full.h5')
    
    print("\n✅ Training complete!")
    print(f"TFLite model: model/waste_efficientnet.tflite")
    print(f"Model info: model/waste_efficientnet_info.json")

if __name__ == "__main__":
    main()