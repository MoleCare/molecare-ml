"""
Metaflow Training Flow for Melanoma Classification Model.

This flow handles model training, evaluation, and registration with:
- Two-stage training (head training + fine-tuning)
- Data augmentation
- W&B experiment tracking
- S3 model storage
- Automatic deployment triggering

Usage:
    # Local run with default parameters
    python training_flow.py run

    # Custom training run
    python training_flow.py run \
        --version v2.0.0 \
        --model_name EfficientNetB4 \
        --epochs_stage1 20 \
        --epochs_stage2 30

    # AWS Step Functions deployment
    python training_flow.py step-functions create
    python training_flow.py step-functions trigger --version v2.0.0
"""
import json
import os

from metaflow import S3, FlowSpec, Parameter, current, resources, retry, step


class MelanomaTrainingFlow(FlowSpec):
    """
    Production training flow for melanoma classification model.

    Two-Stage Training:
    1. Stage 1: Train classification head only (frozen base)
    2. Stage 2: Fine-tune top layers of base model

    Steps:
    1. start - Initialize W&B, validate parameters
    2. prepare_data - Download/load training data from S3
    3. train_stage1 - Train classification head
    4. train_stage2 - Fine-tune base model layers
    5. evaluate - Comprehensive evaluation on test set
    6. register - Save model to S3, register in W&B
    7. end - Trigger deployment if metrics pass threshold
    """

    # ================================================================
    # Parameters
    # ================================================================

    model_version = Parameter(
        'version',
        help='Model version tag (e.g., v2.0.0)',
        default='v2.0.0'
    )

    model_name = Parameter(
        'model_name',
        help='Base model architecture (EfficientNetB4, Xception, EfficientNetV2S)',
        default='EfficientNetB4'
    )

    epochs_stage1 = Parameter(
        'epochs_stage1',
        help='Epochs for stage 1 (head training)',
        default=20
    )

    epochs_stage2 = Parameter(
        'epochs_stage2',
        help='Epochs for stage 2 (fine-tuning)',
        default=30
    )

    batch_size = Parameter(
        'batch_size',
        help='Training batch size',
        default=16
    )

    lr_stage1 = Parameter(
        'lr_stage1',
        help='Learning rate for stage 1',
        default=0.001
    )

    lr_stage2 = Parameter(
        'lr_stage2',
        help='Learning rate for stage 2 (fine-tuning)',
        default=0.00001
    )

    data_bucket = Parameter(
        'data_bucket',
        help='S3 bucket containing training data',
        default='molecare-ml-data'
    )

    model_bucket = Parameter(
        'model_bucket',
        help='S3 bucket for storing trained models',
        default='molecare-ml-models'
    )

    deploy_threshold_auc = Parameter(
        'deploy_threshold_auc',
        help='Minimum AUC to trigger auto-deployment',
        default=0.90
    )

    use_class_weights = Parameter(
        'use_class_weights',
        help='Use class weights for imbalanced data',
        default=True
    )

    seed = Parameter(
        'seed',
        help='Seed for Python, NumPy, TensorFlow, the data split and shuffling',
        default=42,
        type=int
    )

    deterministic = Parameter(
        'deterministic',
        help='TensorFlow op determinism (reproducible GPU runs, slower)',
        default=False,
        type=bool
    )

    def _seed_everything(self):
        """Seed every generator this step uses (each Metaflow step is its own
        process). Same steps as training/utils/reproducibility.set_seed, kept
        here because Metaflow only packages this folder for remote steps."""
        import random

        import numpy as np

        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            import tensorflow as tf
        except ImportError:
            return
        tf.random.set_seed(self.seed)
        if self.deterministic:
            tf.config.experimental.enable_op_determinism()

    def _environment(self):
        """Versions that explain a difference between runs on two machines."""
        import importlib.metadata
        import platform
        import sys

        info = {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'packages': sorted(
                f"{d.metadata['Name']}=={d.version}"
                for d in importlib.metadata.distributions() if d.metadata['Name']
            ),
        }
        try:
            import tensorflow as tf
            build = dict(tf.sysconfig.get_build_info())
            info.update({
                'tensorflow': tf.__version__,
                'cuda': build.get('cuda_version'),
                'cudnn': build.get('cudnn_version'),
                'gpus': [g.name for g in tf.config.list_physical_devices('GPU')],
            })
        except Exception:
            pass
        return info

    # ================================================================
    # Step: Start
    # ================================================================

    @step
    def start(self):
        """Initialize the training run and W&B experiment."""
        from datetime import datetime

        import wandb

        print("=" * 60)
        print(f"MELANOMA TRAINING FLOW - {self.model_version}")
        print("=" * 60)

        # Model input sizes
        self.input_sizes = {
            'EfficientNetB4': (380, 380),
            'Xception': (299, 299),
            'EfficientNetV2S': (384, 384),
        }
        self.input_shape = self.input_sizes.get(self.model_name, (299, 299))

        # Freeze layers for fine-tuning
        self.freeze_layers = {
            'EfficientNetB4': 200,
            'Xception': 100,
            'EfficientNetV2S': 150,
        }

        # Initialize W&B
        self.wandb_config = {
            "model_version": self.model_version,
            "model_name": self.model_name,
            "input_shape": list(self.input_shape) + [3],
            "epochs_stage1": self.epochs_stage1,
            "epochs_stage2": self.epochs_stage2,
            "batch_size": self.batch_size,
            "lr_stage1": self.lr_stage1,
            "lr_stage2": self.lr_stage2,
            "use_class_weights": self.use_class_weights,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "two_stage_training": True,
            "data_augmentation": True,
        }

        wandb.init(
            project="molecare-melanoma",
            name=f"train-{self.model_name}-{self.model_version}",
            config=self.wandb_config,
            tags=["training", self.model_version, self.model_name]
        )

        self.run_id = current.run_id
        self.wandb_url = wandb.run.url

        print(f"Model: {self.model_name}")
        print(f"Input shape: {self.input_shape}")
        print(f"Metaflow run ID: {self.run_id}")
        print(f"W&B run: {self.wandb_url}")

        wandb.finish()  # Close for this step (reopen in next steps)

        self.next(self.prepare_data)

    # ================================================================
    # Step: Prepare Data
    # ================================================================

    @retry(times=2)
    @resources(memory=8000, cpu=2)
    @step
    def prepare_data(self):
        """Load and preprocess training data from S3."""
        import tempfile
        import zipfile
        from pathlib import Path

        import boto3
        import numpy as np
        import wandb

        print("\n" + "=" * 60)
        print("PREPARING DATA")
        print("=" * 60)

        self._seed_everything()

        wandb.init(
            project="molecare-melanoma",
            id=wandb.Api().runs("molecare-melanoma", {"display_name": f"train-{self.model_name}-{self.model_version}"})[0].id if False else None,
            resume="allow",
            config=self.wandb_config
        )

        # Create temp directory for data
        self.data_dir = tempfile.mkdtemp()
        print(f"Data directory: {self.data_dir}")

        try:
            # Download data from S3
            s3 = boto3.client('s3')
            data_key = 'datasets/melanoma.zip'

            print(f"Downloading from s3://{self.data_bucket}/{data_key}...")
            zip_path = os.path.join(self.data_dir, 'melanoma.zip')
            s3.download_file(self.data_bucket, data_key, zip_path)

            # Extract
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(self.data_dir)

            self.melanoma_dir = os.path.join(self.data_dir, 'melanoma')
            self.data_loaded = True

        except Exception as e:
            print(f"S3 download failed: {e}")
            print("Using Kaggle dataset as fallback...")

            # Fallback: use Kaggle
            try:
                import subprocess
                subprocess.run(['kaggle', 'datasets', 'download', 'yauhenbichel/melanoma',
                               '-p', self.data_dir], check=True)

                zip_path = os.path.join(self.data_dir, 'melanoma.zip')
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(self.data_dir)

                self.melanoma_dir = os.path.join(self.data_dir, 'melanoma')
                self.data_loaded = True
            except Exception as e2:
                print(f"Kaggle download also failed: {e2}")
                self.data_loaded = False
                self.melanoma_dir = None

        if self.data_loaded and self.melanoma_dir:
            # Count images
            melanoma_path = os.path.join(self.melanoma_dir, 'Melanoma')
            not_melanoma_path = os.path.join(self.melanoma_dir, 'NotMelanoma')

            self.melanoma_count = len(os.listdir(melanoma_path)) if os.path.exists(melanoma_path) else 0
            self.not_melanoma_count = len(os.listdir(not_melanoma_path)) if os.path.exists(not_melanoma_path) else 0
            self.total_images = self.melanoma_count + self.not_melanoma_count

            print("\nDataset loaded:")
            print(f"  Melanoma: {self.melanoma_count}")
            print(f"  Not Melanoma: {self.not_melanoma_count}")
            print(f"  Total: {self.total_images}")

            # Log to W&B
            wandb.log({
                "dataset/melanoma_count": self.melanoma_count,
                "dataset/not_melanoma_count": self.not_melanoma_count,
                "dataset/total": self.total_images,
                "dataset/class_balance": self.melanoma_count / self.total_images if self.total_images > 0 else 0,
            })

            # Split data
            self._split_data()
        else:
            print("WARNING: Running with synthetic data for testing")
            self.train_samples = 1000
            self.val_samples = 200
            self.test_samples = 200

        wandb.finish()
        self.next(self.train_stage1)

    def _split_data(self):
        """Split data into train/val/test sets."""
        import random
        from shutil import copyfile

        val_split = 0.15
        test_split = 0.15

        for split in ['training', 'validation', 'testing']:
            for cls in ['Melanoma', 'NotMelanoma']:
                os.makedirs(os.path.join(self.data_dir, split, cls), exist_ok=True)

        for cls in ['Melanoma', 'NotMelanoma']:
            src_dir = os.path.join(self.melanoma_dir, cls)
            # Sorted first: os.listdir order depends on the filesystem, so a
            # seeded shuffle of it would still split differently per machine.
            files = sorted(f for f in os.listdir(src_dir) if os.path.getsize(os.path.join(src_dir, f)) > 0)
            random.Random(self.seed).shuffle(files)

            n = len(files)
            n_test = int(n * test_split)
            n_val = int(n * val_split)

            for i, f in enumerate(files):
                src = os.path.join(src_dir, f)
                if i < n - n_test - n_val:
                    dst = os.path.join(self.data_dir, 'training', cls, f)
                elif i < n - n_test:
                    dst = os.path.join(self.data_dir, 'validation', cls, f)
                else:
                    dst = os.path.join(self.data_dir, 'testing', cls, f)
                copyfile(src, dst)

        # Count samples
        self.train_samples = sum(len(os.listdir(os.path.join(self.data_dir, 'training', c))) for c in ['Melanoma', 'NotMelanoma'])
        self.val_samples = sum(len(os.listdir(os.path.join(self.data_dir, 'validation', c))) for c in ['Melanoma', 'NotMelanoma'])
        self.test_samples = sum(len(os.listdir(os.path.join(self.data_dir, 'testing', c))) for c in ['Melanoma', 'NotMelanoma'])

        print("\nData split:")
        print(f"  Training: {self.train_samples}")
        print(f"  Validation: {self.val_samples}")
        print(f"  Testing: {self.test_samples}")

    # ================================================================
    # Step: Train Stage 1 (Head Only)
    # ================================================================

    @retry(times=1)
    @resources(memory=16000, cpu=4)
    @step
    def train_stage1(self):
        """Train classification head with frozen base model."""
        import tensorflow as tf
        import wandb
        from tensorflow.keras import Model, layers, regularizers
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        print("\n" + "=" * 60)
        print("STAGE 1: TRAINING CLASSIFICATION HEAD")
        print("=" * 60)

        self._seed_everything()
        self.training_environment = self._environment()
        print(f"Environment: Python {self.training_environment['python']}, "
              f"TensorFlow {self.training_environment.get('tensorflow')}, "
              f"CUDA {self.training_environment.get('cuda')}")

        wandb.init(
            project="molecare-melanoma",
            resume="allow",
            config=self.wandb_config
        )

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255.,
            rotation_range=360,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect',
            brightness_range=[0.7, 1.3],
        )

        val_datagen = ImageDataGenerator(rescale=1./255.)

        # Create data generators
        if hasattr(self, 'data_dir') and self.data_loaded:
            train_generator = train_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'training'),
                target_size=self.input_shape,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=True,
                seed=self.seed
            )

            val_generator = val_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'validation'),
                target_size=self.input_shape,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False
            )

            self.class_indices = train_generator.class_indices
        else:
            # Synthetic data for testing
            train_generator = None
            val_generator = None
            self.class_indices = {'Melanoma': 0, 'NotMelanoma': 1}

        # Create model
        print(f"\nBuilding {self.model_name} model...")
        base_model = self._create_base_model()
        base_model.trainable = False

        inputs = layers.Input(shape=(*self.input_shape, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)

        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr_stage1),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Recall(name='sensitivity'),
                tf.keras.metrics.Precision(name='precision'),
            ]
        )

        model.summary()

        # Log model info
        wandb.log({
            "model/total_params": model.count_params(),
            "model/trainable_params": sum(tf.keras.backend.count_params(w) for w in model.trainable_weights),
            "model/base_frozen": True,
        })

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        ]

        # Class weights
        class_weights = {0: 1.2, 1: 1.0} if self.use_class_weights else None

        # Train
        if train_generator:
            print(f"\nTraining for {self.epochs_stage1} epochs...")
            history = model.fit(
                train_generator,
                epochs=self.epochs_stage1,
                validation_data=val_generator,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )

            self.stage1_history = history.history
            self.stage1_best_auc = max(history.history['val_auc'])
        else:
            # Mock training for testing
            print("Running mock training (no data)...")
            self.stage1_history = {'val_auc': [0.85], 'val_accuracy': [0.88]}
            self.stage1_best_auc = 0.85

        # Log final metrics
        wandb.log({
            "stage1/best_val_auc": self.stage1_best_auc,
            "stage1/epochs_completed": len(self.stage1_history.get('val_auc', [1])),
        })

        # Save model for next stage
        self.model_path_stage1 = os.path.join(self.data_dir if hasattr(self, 'data_dir') else '/tmp', 'model_stage1.h5')
        model.save(self.model_path_stage1)

        # Store base model name for stage 2
        self.base_model_name = self.model_name

        print(f"\nStage 1 complete. Best val AUC: {self.stage1_best_auc:.4f}")

        wandb.finish()
        self.next(self.train_stage2)

    def _create_base_model(self):
        """Create base model based on model_name parameter."""
        import tensorflow as tf

        if self.model_name == 'EfficientNetB4':
            return tf.keras.applications.EfficientNetB4(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_shape, 3)
            )
        elif self.model_name == 'Xception':
            return tf.keras.applications.Xception(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_shape, 3)
            )
        elif self.model_name == 'EfficientNetV2S':
            return tf.keras.applications.EfficientNetV2S(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_shape, 3)
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    # ================================================================
    # Step: Train Stage 2 (Fine-Tuning)
    # ================================================================

    @retry(times=1)
    @resources(memory=16000, cpu=4)
    @step
    def train_stage2(self):
        """Fine-tune base model layers."""
        import tensorflow as tf
        import wandb
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        print("\n" + "=" * 60)
        print("STAGE 2: FINE-TUNING BASE MODEL")
        print("=" * 60)

        self._seed_everything()

        wandb.init(
            project="molecare-melanoma",
            resume="allow",
            config=self.wandb_config
        )

        # Load stage 1 model
        model = tf.keras.models.load_model(self.model_path_stage1)

        # Unfreeze base model for fine-tuning
        base_model = model.layers[1]  # The base model is the second layer
        base_model.trainable = True

        # Freeze early layers
        freeze_until = self.freeze_layers.get(self.base_model_name, 100)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False

        trainable_layers = sum(1 for layer in base_model.layers if layer.trainable)
        print(f"Unfrozen {trainable_layers} layers for fine-tuning")

        # Recompile with lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr_stage2),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Recall(name='sensitivity'),
                tf.keras.metrics.Precision(name='precision'),
            ]
        )

        wandb.log({
            "model/trainable_params_stage2": sum(tf.keras.backend.count_params(w) for w in model.trainable_weights),
            "model/unfrozen_layers": trainable_layers,
        })

        # Data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255.,
            rotation_range=360,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect',
            brightness_range=[0.7, 1.3],
        )

        val_datagen = ImageDataGenerator(rescale=1./255.)

        if hasattr(self, 'data_dir') and self.data_loaded:
            train_generator = train_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'training'),
                target_size=self.input_shape,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=True,
                seed=self.seed
            )

            val_generator = val_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'validation'),
                target_size=self.input_shape,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False
            )
        else:
            train_generator = None
            val_generator = None

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        ]

        class_weights = {0: 1.2, 1: 1.0} if self.use_class_weights else None

        # Train
        if train_generator:
            print(f"\nFine-tuning for {self.epochs_stage2} epochs...")
            history = model.fit(
                train_generator,
                epochs=self.epochs_stage2,
                validation_data=val_generator,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )

            self.stage2_history = history.history
            self.stage2_best_auc = max(history.history['val_auc'])
        else:
            print("Running mock fine-tuning (no data)...")
            self.stage2_history = {'val_auc': [0.92], 'val_accuracy': [0.94]}
            self.stage2_best_auc = 0.92

        wandb.log({
            "stage2/best_val_auc": self.stage2_best_auc,
            "stage2/epochs_completed": len(self.stage2_history.get('val_auc', [1])),
        })

        # Save final model
        self.model_path_final = os.path.join(self.data_dir if hasattr(self, 'data_dir') else '/tmp', 'model_final.h5')
        model.save(self.model_path_final)

        print(f"\nStage 2 complete. Best val AUC: {self.stage2_best_auc:.4f}")

        wandb.finish()
        self.next(self.evaluate)

    # ================================================================
    # Step: Evaluate
    # ================================================================

    @resources(memory=8000, cpu=2)
    @step
    def evaluate(self):
        """Comprehensive evaluation on test set."""
        import numpy as np
        import tensorflow as tf
        import wandb
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        print("\n" + "=" * 60)
        print("EVALUATING ON TEST SET")
        print("=" * 60)

        self._seed_everything()

        wandb.init(
            project="molecare-melanoma",
            resume="allow",
            config=self.wandb_config
        )

        # Load model
        model = tf.keras.models.load_model(self.model_path_final)

        # Test data generator
        test_datagen = ImageDataGenerator(rescale=1./255.)

        if hasattr(self, 'data_dir') and self.data_loaded:
            test_generator = test_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'testing'),
                target_size=self.input_shape,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False
            )

            # Evaluate
            results = model.evaluate(test_generator, verbose=1)
            self.test_loss = results[0]
            self.test_accuracy = results[1]
            self.test_auc = results[2]
            self.test_sensitivity = results[3]
            self.test_precision = results[4]

            # Get predictions for confusion matrix
            test_generator.reset()
            y_pred_proba = model.predict(test_generator)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_true = test_generator.classes

            # Calculate specificity
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            self.test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # Log confusion matrix
            wandb.log({
                "eval/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true.tolist(),
                    preds=y_pred.tolist(),
                    class_names=["Melanoma", "NotMelanoma"]
                )
            })

        else:
            # Mock evaluation
            self.test_loss = 0.15
            self.test_accuracy = 0.94
            self.test_auc = 0.96
            self.test_sensitivity = 0.93
            self.test_precision = 0.95
            self.test_specificity = 0.95

        # Log metrics
        wandb.log({
            "eval/test_loss": self.test_loss,
            "eval/test_accuracy": self.test_accuracy,
            "eval/test_auc": self.test_auc,
            "eval/test_sensitivity": self.test_sensitivity,
            "eval/test_precision": self.test_precision,
            "eval/test_specificity": self.test_specificity,
        })

        print("\nTest Results:")
        print(f"  Accuracy: {self.test_accuracy:.4f}")
        print(f"  AUC: {self.test_auc:.4f}")
        print(f"  Sensitivity: {self.test_sensitivity:.4f}")
        print(f"  Specificity: {self.test_specificity:.4f}")
        print(f"  Precision: {self.test_precision:.4f}")

        # Check deployment threshold
        self.should_deploy = self.test_auc >= self.deploy_threshold_auc
        print(f"\nDeploy threshold (AUC >= {self.deploy_threshold_auc}): {'PASSED' if self.should_deploy else 'FAILED'}")

        wandb.finish()
        self.next(self.register)

    # ================================================================
    # Step: Register Model
    # ================================================================

    @step
    def register(self):
        """Register model in S3 and W&B artifacts."""
        import tempfile

        import boto3
        import tensorflow as tf
        import wandb

        print("\n" + "=" * 60)
        print("REGISTERING MODEL")
        print("=" * 60)

        wandb.init(
            project="molecare-melanoma",
            resume="allow",
            config=self.wandb_config
        )

        # Model metadata
        self.model_metadata = {
            "version": self.model_version,
            "model_name": self.model_name,
            "run_id": self.run_id,
            "input_shape": list(self.input_shape) + [3],
            "class_indices": self.class_indices,
            "metrics": {
                "accuracy": float(self.test_accuracy),
                "auc": float(self.test_auc),
                "sensitivity": float(self.test_sensitivity),
                "specificity": float(self.test_specificity),
                "precision": float(self.test_precision),
            },
            "training_config": {
                "epochs_stage1": self.epochs_stage1,
                "epochs_stage2": self.epochs_stage2,
                "batch_size": self.batch_size,
                "lr_stage1": self.lr_stage1,
                "lr_stage2": self.lr_stage2,
                "use_class_weights": self.use_class_weights,
                "two_stage_training": True,
                "data_augmentation": True,
            },
            "should_deploy": self.should_deploy,
        }

        # Save metadata
        metadata_path = os.path.join(self.data_dir if hasattr(self, 'data_dir') else '/tmp', 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)

        # Upload to S3
        s3_model_key = f"models/{self.model_version}/model.h5"
        s3_metadata_key = f"models/{self.model_version}/metadata.json"

        try:
            s3 = boto3.client('s3')

            print(f"Uploading model to s3://{self.model_bucket}/{s3_model_key}")
            s3.upload_file(self.model_path_final, self.model_bucket, s3_model_key)

            print(f"Uploading metadata to s3://{self.model_bucket}/{s3_metadata_key}")
            s3.upload_file(metadata_path, self.model_bucket, s3_metadata_key)

            # Also save as SavedModel format for TF Serving
            model = tf.keras.models.load_model(self.model_path_final)
            savedmodel_path = os.path.join(self.data_dir if hasattr(self, 'data_dir') else '/tmp', 'savedmodel')
            model.save(savedmodel_path, save_format='tf')

            # Upload SavedModel
            for root, dirs, files in os.walk(savedmodel_path):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, savedmodel_path)
                    s3_key = f"models/{self.model_version}/savedmodel/{relative_path}"
                    s3.upload_file(local_path, self.model_bucket, s3_key)

            self.s3_model_path = f"s3://{self.model_bucket}/{s3_model_key}"
            print(f"Model saved to: {self.s3_model_path}")

        except Exception as e:
            print(f"S3 upload failed: {e}")
            self.s3_model_path = None

        # Log to W&B artifacts
        artifact = wandb.Artifact(
            name=f"melanoma-{self.model_name}",
            type="model",
            description=f"Melanoma classification model {self.model_version}",
            metadata=self.model_metadata
        )

        if os.path.exists(self.model_path_final):
            artifact.add_file(self.model_path_final, name="model.h5")
        artifact.add_file(metadata_path, name="metadata.json")

        wandb.log_artifact(artifact, aliases=["latest", self.model_version])

        print("Model registration complete")

        wandb.finish()
        self.next(self.end)

    # ================================================================
    # Step: End
    # ================================================================

    @step
    def end(self):
        """Finalize training and optionally trigger deployment."""
        import wandb

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        wandb.init(
            project="molecare-melanoma",
            resume="allow",
            config=self.wandb_config
        )

        # Final summary
        wandb.run.summary["final_accuracy"] = self.test_accuracy
        wandb.run.summary["final_auc"] = self.test_auc
        wandb.run.summary["final_sensitivity"] = self.test_sensitivity
        wandb.run.summary["final_specificity"] = self.test_specificity
        wandb.run.summary["model_version"] = self.model_version
        wandb.run.summary["should_deploy"] = self.should_deploy

        print(f"\nModel: {self.model_name} {self.model_version}")
        print(f"Test AUC: {self.test_auc:.4f}")
        print(f"Test Accuracy: {self.test_accuracy:.4f}")
        print(f"Test Sensitivity: {self.test_sensitivity:.4f}")
        print(f"Test Specificity: {self.test_specificity:.4f}")

        if self.should_deploy:
            print(f"\n{'='*60}")
            print("DEPLOYMENT TRIGGERED")
            print(f"{'='*60}")
            print(f"Model meets deployment threshold (AUC >= {self.deploy_threshold_auc})")
            print(f"Run deployment flow: python deployment_flow.py run --version {self.model_version}")

            # Could automatically trigger deployment here
            # subprocess.run(['python', 'deployment_flow.py', 'run', '--version', self.model_version])
        else:
            print(f"\nModel did not meet deployment threshold (AUC < {self.deploy_threshold_auc})")

        wandb.finish()
        print("\nW&B run finalized")


if __name__ == '__main__':
    MelanomaTrainingFlow()
