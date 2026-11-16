
"""
model_contract.py

Purpose
-------
This module provides a standardized interface between our FastAPI backend
and the externally trained EEG blink-detection model.

The rest of the backend should not need to know whether the supplied model
uses PyTorch, TensorFlow, scikit-learn, or another framework.

Our backend will:
    1. Load the model once.
    2. Pass it a validated EEG window.
    3. Receive its prediction.
    4. Convert that prediction into a standardized BlinkPrediction object.

IMPORTANT MODEL PLUG-IN CONVENTION
----------------------------------
The Python module specified by MODEL_MODULE must expose the function
specified by MODEL_FUNCTION.

There are two supported ways to provide the model:

A. Module with a load_model() function:
       def load_model():
           # Load and return the trained model.
           return model

       def predict_blink(model, signal_window):
           # Run inference using the loaded model.
           return model_output

B. Module without a load_model() function:
       def predict_blink(signal_window):
           # The module manages its model internally.
           return model_output

The prediction function receives a one-dimensional NumPy array containing
chronologically ordered EEG samples.

The model-specific module is responsible for reshaping this array if its
framework requires a different input shape.
"""

import importlib
import logging
import threading
import time

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


# Create a logger specifically for this module.
# The application can configure logging centrally in app.py.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CUSTOM EXCEPTIONS
# ---------------------------------------------------------------------

class ModelContractError(Exception):
    """
    Base exception for errors related to the model contract.

    Using a dedicated exception type allows app.py to distinguish model
    integration errors from unrelated application errors.
    """
    pass


class ModelLoadingError(ModelContractError):
    """
    Raised when the model module, model loader, or prediction function
    cannot be loaded or initialized.
    """
    pass


class ModelInputError(ModelContractError):
    """
    Raised when the EEG window does not satisfy the input requirements.

    Examples:
        - Empty input.
        - Incorrect number of dimensions.
        - Invalid numerical values such as NaN or infinity.
        - Unexpected number of samples.
    """
    pass


class ModelOutputError(ModelContractError):
    """
    Raised when the trained model returns an unsupported or invalid output.

    This prevents the backend from silently interpreting an unexpected
    model result as a blink or a non-blink.
    """
    pass


class ModelInferenceError(ModelContractError):
    """
    Raised when the prediction function fails during inference.
    """
    pass


# ---------------------------------------------------------------------
# STANDARDIZED PREDICTION RESULT
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class BlinkPrediction:
    """
    Represents one standardized blink-detection result.

    Attributes
    ----------
    blink_detected:
        True if the model classified the input window as containing a blink.
        False otherwise.

    label:
        Human-readable classification: "blink" or "no_blink".

    score:
        Optional numerical score associated with the blink prediction.
        For probability-based outputs, this is the blink probability.
        For binary-label outputs, it is None because a confidence score
        cannot safely be inferred from a class label alone.

    output_type:
        Identifies the original output format used by the model:
        "label", "probability", "probabilities", or "logit".

    inference_ms:
        Time taken to execute and interpret this prediction, in milliseconds.
    """

    blink_detected: bool
    label: str
    score: Optional[float]
    output_type: str
    inference_ms: float

    def to_dict(self) -> dict:
        """
        Convert the prediction object into a JSON-compatible dictionary.

        Why do we need this?
        --------------------
        FastAPI communicates with our frontend through JSON messages.

        A Python dataclass is convenient inside our backend, but it should
        be converted into a regular dictionary before being sent through
        a WebSocket.

        Example result:
            {
                "blink_detected": True,
                "label": "blink",
                "score": 0.92,
                "output_type": "probability",
                "inference_ms": 3.4
            }

        The dictionary contains only standard Python types, making it
        straightforward to serialize.
        """

        return {
            "blink_detected": self.blink_detected,
            "label": self.label,
            "score": self.score,
            "output_type": self.output_type,
            "inference_ms": round(self.inference_ms, 3),
        }


# ---------------------------------------------------------------------
# MODEL ADAPTER
# ---------------------------------------------------------------------

class BlinkModel:
    """
    Provides a consistent interface for loading and using the trained
    EEG blink-detection model.

    This class separates model-specific implementation details from
    the rest of our backend.

    For example, app.py will eventually call:

        prediction = blink_model.predict(signal_window)

    It will not need to know how the model was trained or which ML
    framework was used.

    Parameters
    ----------
    settings:
        The settings object created in config.py.

        It must contain:
            MODEL_MODULE
            MODEL_FUNCTION
            BLINK_THRESHOLD

    output_type:
        Specifies the actual format returned by the trained model.

        Supported values:
            "label"
            "probability"
            "probabilities"
            "logit"

        This must be selected after inspecting the supplied model.

    expected_samples:
        Optional exact number of samples expected in each input window.

        For example, if the model requires 250 samples per prediction,
        expected_samples should be 250.

        Leave it as None until we know the model's required window size.

    blink_class_index:
        Used only when output_type="probabilities".

        Specifies which position in the probability vector corresponds
        to the blink class.

        For example, if:
            [0.15, 0.85]

        and blink_class_index=1, the blink probability is 0.85.

    positive_label:
        Used only when output_type="label".

        Specifies which returned class label represents a blink.

        For example:
            positive_label=1

        means that a returned label of 1 represents a blink.

    """

    def __init__(
        self,
        settings,
        output_type: str,
        expected_samples: Optional[int] = None,
        blink_class_index: int = 1,
        positive_label: Any = 1,
    ):

        # Read the model module and function names from config.py.
        # These identify where the trained model integration lives.
        self.module_name = settings.MODEL_MODULE
        self.function_name = settings.MODEL_FUNCTION

        # Read the blink threshold from config.py.
        # This threshold is used only for probability and logit outputs.
        self.threshold = float(settings.BLINK_THRESHOLD)

        # Store the expected output format.
        # We deliberately do not guess this from the returned value.
        self.output_type = output_type.lower()

        # Store the expected input window size.
        # None means that we validate the input dimensions and values,
        # but do not enforce a particular number of samples.
        self.expected_samples = expected_samples

        # Store the index representing the blink class in a probability
        # vector, when the model returns multiple class probabilities.
        self.blink_class_index = blink_class_index

        # Store the class label that represents a blink when the model
        # returns a direct class label.
        self.positive_label = positive_label

        # These attributes are initialized later by load().
        self.module = None
        self.model = None
        self.predict_function = None

        # Indicates whether the model has been successfully initialized.
        self.loaded = False

        # A lock prevents two prediction calls from using the model
        # simultaneously. This is useful for models that are not
        # thread-safe.
        self.inference_lock = threading.Lock()

        # Validate the contract configuration immediately.
        self._validate_configuration()

    # -----------------------------------------------------------------
    # CONFIGURATION VALIDATION
    # -----------------------------------------------------------------

    def _validate_configuration(self):
        """
        Validate the configuration supplied when creating BlinkModel.

        Why is this necessary?
        ----------------------
        Configuration errors should be detected before acquisition starts.

        For example, if the model output type is misspelled as
        "probablity", the application should fail immediately rather
        than wait until the first EEG prediction.

        This function checks:
            1. Whether the output format is supported.
            2. Whether the threshold is between 0 and 1.
            3. Whether the expected sample count is positive.
            4. Whether the blink class index is non-negative.
        """

        supported_types = {
            "label",
            "probability",
            "probabilities",
            "logit",
        }

        if self.output_type not in supported_types:
            raise ModelContractError(
                f"Unsupported output_type '{self.output_type}'. "
                f"Choose one of: {sorted(supported_types)}"
            )

        if not 0.0 <= self.threshold <= 1.0:
            raise ModelContractError(
                "BLINK_THRESHOLD must be between 0 and 1."
            )

        if self.expected_samples is not None:
            if self.expected_samples <= 0:
                raise ModelContractError(
                    "expected_samples must be a positive integer."
                )

        if self.blink_class_index < 0:
            raise ModelContractError(
                "blink_class_index cannot be negative."
            )

    # -----------------------------------------------------------------
    # MODEL LOADING
    # -----------------------------------------------------------------

    def load(self):
        """
        Import the trained model module and initialize the model.

        Why do we need this function?
        -----------------------------
        Loading a trained model can be relatively expensive.

        We should not load the model every time a new EEG batch arrives.
        Instead, app.py should call load() once when the application starts.

        This function performs the following operations:

            1. Import the module specified by MODEL_MODULE.
            2. Look for an optional load_model() function.
            3. If load_model() exists, execute it once and store its result.
            4. Retrieve the prediction function specified by MODEL_FUNCTION.
            5. Verify that the prediction function is callable.
            6. Mark the model as loaded.

        Two supported module arrangements:

        Arrangement A:
            The module has load_model() and predict_blink(model, window).

        Arrangement B:
            The module has predict_blink(window), and manages its model
            internally.

        The prediction function is not called here. It is called later
        by predict() whenever a new EEG window is ready.

        Returns
        -------
        self:
            Returning the object allows convenient initialization:

                blink_model = BlinkModel(...).load()
        """

        if self.loaded:
            logger.info("Blink model is already loaded.")
            return self

        try:
            # Dynamically import the Python module.
            # For example, MODEL_MODULE="blink_model" imports blink_model.py.
            self.module = importlib.import_module(self.module_name)

            # Retrieve the configured prediction function from that module.
            self.predict_function = getattr(
                self.module,
                self.function_name,
                None,
            )

            # Verify that the requested prediction function actually exists.
            if not callable(self.predict_function):
                raise ModelLoadingError(
                    f"Function '{self.function_name}' was not found "
                    f"or is not callable in module '{self.module_name}'."
                )

            # Check whether the module provides a dedicated model loader.
            # getattr() returns None if the function does not exist.
            model_loader = getattr(self.module, "load_model", None)

            if callable(model_loader):

                # Load the trained model exactly once.
                # The returned object is stored for future predictions.
                self.model = model_loader()

                # A loader that returns None is considered invalid here.
                if self.model is None:
                    raise ModelLoadingError(
                        "load_model() returned None. "
                        "It must return the initialized model object."
                    )

                # The prediction function must now accept:
                # predict_function(model, signal_window)
                self.uses_loaded_model_argument = True

            else:
                # No separate loader exists.
                # The prediction function must manage its model internally
                # and accept only:
                # predict_function(signal_window)
                self.model = None
                self.uses_loaded_model_argument = False

            self.loaded = True

            logger.info(
                "Blink model loaded successfully from module '%s'.",
                self.module_name,
            )

            return self

        except ModelLoadingError:
            # Preserve errors that already have a meaningful explanation.
            raise

        except Exception as exc:
            # Convert import errors, missing dependencies, file errors,
            # and initialization failures into a consistent exception.
            raise ModelLoadingError(
                f"Could not load the blink model: {exc}"
            ) from exc

    # -----------------------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------------------

    def validate_input(self, signal_window: Any) -> np.ndarray:
        """
        Validate and prepare one EEG window before inference.

        Expected input
        --------------
        A one-dimensional sequence of chronological A4 samples.

        Example:
            [512, 514, 510, 520, 530, ...]

        Why one-dimensional?
        --------------------
        We are acquiring one analog EEG channel: A4.

        Therefore, one window represents one sequence of EEG samples.

        If the model needs a shape such as (1, 1, 250), the model-specific
        prediction function should reshape this validated one-dimensional
        window internally.

        Operations performed:
            1. Convert the input into a NumPy array.
            2. Convert the values to float32.
            3. Verify that the array is one-dimensional.
            4. Reject empty arrays.
            5. Check the expected number of samples, if configured.
            6. Reject NaN and infinite values.
            7. Return a contiguous NumPy array.

        The function does not filter, normalize, or resample the signal.
        Those operations must reproduce the trained model's preprocessing.

        Returns
        -------
        np.ndarray:
            A validated one-dimensional float32 EEG signal window.
        """

        try:
            # Convert lists or other numerical sequences into a NumPy array.
            # float32 is commonly used by ML models and reduces memory use.
            signal = np.asarray(signal_window, dtype=np.float32)

        except (TypeError, ValueError) as exc:
            raise ModelInputError(
                "The EEG window must contain numerical values."
            ) from exc

        # A one-dimensional array has the form:
        # [sample_1, sample_2, sample_3, ...]
        #
        # Reject two-dimensional arrays because this contract expects
        # one channel, not a matrix of multiple channels.
        if signal.ndim != 1:
            raise ModelInputError(
                f"Expected a one-dimensional EEG window, "
                f"but received shape {signal.shape}."
            )

        # A model cannot make a meaningful prediction from zero samples.
        if signal.size == 0:
            raise ModelInputError(
                "The EEG window is empty."
            )

        # If the model requires a fixed window size, enforce it here.
        # For example, expected_samples=250 means every prediction must
        # receive exactly 250 EEG samples.
        if self.expected_samples is not None:
            if signal.size != self.expected_samples:
                raise ModelInputError(
                    f"Expected {self.expected_samples} samples, "
                    f"but received {signal.size}."
                )

        # NaN and infinity can cause unpredictable behavior in ML models.
        # Reject them before passing the signal to the model.
        if not np.all(np.isfinite(signal)):
            raise ModelInputError(
                "The EEG window contains NaN or infinite values."
            )

        # Ensure that the returned array occupies a contiguous block
        # of memory. This can be useful for numerical libraries.
        return np.ascontiguousarray(signal, dtype=np.float32)

    # -----------------------------------------------------------------
    # OUTPUT NORMALIZATION
    # -----------------------------------------------------------------

    def _normalize_output(
        self,
        raw_output: Any,
        inference_ms: float,
    ) -> BlinkPrediction:
        """
        Convert the trained model's output into a BlinkPrediction.

        Why do we need output normalization?
        -------------------------------------
        Different models represent their predictions differently.

        For example:
            Model A returns: 1
            Model B returns: 0.93
            Model C returns: [0.07, 0.93]
            Model D returns: 2.6 (a logit)

        app.py and the frontend should not need separate logic for each
        possible output format.

        This function interprets the result according to output_type,
        then creates the same BlinkPrediction structure every time.

        Supported formats
        -----------------

        1. label:
            A direct class label, such as 0 or 1.
            positive_label determines which value means blink.

        2. probability:
            A single blink probability between 0 and 1.
            BLINK_THRESHOLD determines the classification.

        3. probabilities:
            A one-dimensional array of class probabilities.
            blink_class_index selects the blink probability.

        4. logit:
            A single raw logit.
            The sigmoid function converts it into a probability.

        Returns
        -------
        BlinkPrediction:
            A standardized prediction object.

        Raises
        ------
        ModelOutputError:
            If the output cannot be interpreted safely.
        """

        # -------------------------------------------------------------
        # CASE 1: DIRECT CLASS LABEL
        # -------------------------------------------------------------

        if self.output_type == "label":

            # A direct class label should contain exactly one value.
            values = np.asarray(raw_output)

            if values.size != 1:
                raise ModelOutputError(
                    "A label output must contain exactly one class value."
                )

            raw_label = values.item()

            # Compare the returned label against the configured positive
            # label. For example, positive_label=1 means class 1 is a blink.
            is_blink = raw_label == self.positive_label

            if not isinstance(is_blink, (bool, np.bool_)):
                raise ModelOutputError(
                    "The returned class label could not be compared "
                    "with positive_label."
                )

            return BlinkPrediction(
                blink_detected=bool(is_blink),
                label="blink" if is_blink else "no_blink",
                score=None,
                output_type=self.output_type,
                inference_ms=inference_ms,
            )

        # -------------------------------------------------------------
        # CASE 2: SINGLE PROBABILITY
        # -------------------------------------------------------------

        if self.output_type == "probability":

            # The model should return one numerical probability.
            values = np.asarray(raw_output, dtype=np.float64)

            if values.size != 1:
                raise ModelOutputError(
                    "A probability output must contain exactly one value."
                )

            probability = float(values.item())

            # A probability must be finite and within [0, 1].
            if not np.isfinite(probability):
                raise ModelOutputError(
                    "The model returned a non-finite probability."
                )

            if not 0.0 <= probability <= 1.0:
                raise ModelOutputError(
                    f"Invalid probability value: {probability}."
                )

            # A blink is detected when its probability reaches the
            # configured threshold.
            is_blink = probability >= self.threshold

            return BlinkPrediction(
                blink_detected=is_blink,
                label="blink" if is_blink else "no_blink",
                score=probability,
                output_type=self.output_type,
                inference_ms=inference_ms,
            )

        # -------------------------------------------------------------
        # CASE 3: VECTOR OF CLASS PROBABILITIES
        # -------------------------------------------------------------

        if self.output_type == "probabilities":

            # Convert the returned class probabilities into a numerical
            # NumPy array.
            probabilities = np.asarray(
                raw_output,
                dtype=np.float64,
            )

            # This contract expects one vector, for example [0.1, 0.9].
            if probabilities.ndim != 1:
                raise ModelOutputError(
                    "Class probabilities must be a one-dimensional vector."
                )

            # Verify that the configured blink index exists.
            if self.blink_class_index >= probabilities.size:
                raise ModelOutputError(
                    f"blink_class_index={self.blink_class_index} is out "
                    f"of range for {probabilities.size} classes."
                )

            # All returned probabilities must be finite and within [0, 1].
            if not np.all(np.isfinite(probabilities)):
                raise ModelOutputError(
                    "The model returned non-finite class probabilities."
                )

            if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
                raise ModelOutputError(
                    "Class probabilities must be between 0 and 1."
                )

            # Extract the probability belonging to the blink class.
            blink_probability = float(
                probabilities[self.blink_class_index]
            )

            # Apply the configured classification threshold.
            is_blink = blink_probability >= self.threshold

            return BlinkPrediction(
                blink_detected=is_blink,
                label="blink" if is_blink else "no_blink",
                score=blink_probability,
                output_type=self.output_type,
                inference_ms=inference_ms,
            )

        # -------------------------------------------------------------
        # CASE 4: SINGLE LOGIT
        # -------------------------------------------------------------

        if self.output_type == "logit":

            # A logit is an unrestricted numerical value.
            # Unlike a probability, it can be negative or positive.
            values = np.asarray(raw_output, dtype=np.float64)

            if values.size != 1:
                raise ModelOutputError(
                    "A logit output must contain exactly one value."
                )

            logit = float(values.item())

            if not np.isfinite(logit):
                raise ModelOutputError(
                    "The model returned a non-finite logit."
                )

            # Convert the logit into a probability using sigmoid:
            #
            # sigmoid(x) = 1 / (1 + exp(-x))
            #
            # This implementation avoids numerical overflow for
            # very large positive or negative logits.
            if logit >= 0:
                probability = 1.0 / (1.0 + np.exp(-logit))
            else:
                exp_value = np.exp(logit)
                probability = exp_value / (1.0 + exp_value)

            # Classify the converted probability using our threshold.
            is_blink = probability >= self.threshold

            return BlinkPrediction(
                blink_detected=is_blink,
                label="blink" if is_blink else "no_blink",
                score=float(probability),
                output_type=self.output_type,
                inference_ms=inference_ms,
            )

        # Configuration validation should prevent reaching this point.
        # This final check protects against unexpected changes.
        raise ModelOutputError(
            f"Unsupported output type: {self.output_type}."
        )

    # -----------------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------------

    def predict(self, signal_window: Any) -> BlinkPrediction:
        """
        Run blink detection on one validated EEG window.

        This is the main public function that app.py will use.

        Workflow
        --------
            1. Verify that the model has been loaded.
            2. Validate the incoming EEG window.
            3. Acquire the inference lock.
            4. Call the trained model's prediction function.
            5. Measure inference duration.
            6. Normalize the model output.
            7. Return a BlinkPrediction object.

        Why use a lock?
        ---------------
        If multiple WebSocket clients or backend tasks request inference
        simultaneously, they could attempt to use the same model at once.

        Some model implementations are not thread-safe. The lock ensures
        that only one prediction call enters the model at a time.

        Why measure inference time?
        ---------------------------
        We want to evaluate real-time performance.

        If inference takes 5 ms, that is useful information when deciding
        how frequently the backend can run predictions.

        Note that inference time is not the same as total end-to-end latency.
        It excludes acquisition, buffering, network transmission, and
        frontend rendering.

        Parameters
        ----------
        signal_window:
            One-dimensional sequence of EEG samples.

        Returns
        -------
        BlinkPrediction:
            Standardized classification result.
        """

        # The model must be loaded before predictions can be made.
        if not self.loaded:
            raise ModelInferenceError(
                "The blink model has not been loaded. Call load() first."
            )

        # Validate and convert the input before entering the model.
        signal = self.validate_input(signal_window)

        try:

            # Prevent simultaneous use of the same model instance.
            with self.inference_lock:

                # Start a high-resolution timer immediately before inference.
                start_time = time.perf_counter()

                # If a separate load_model() function was provided,
                # pass the loaded model and signal window.
                if self.uses_loaded_model_argument:
                    raw_output = self.predict_function(
                        self.model,
                        signal,
                    )

                # Otherwise, the module manages its model internally
                # and the prediction function receives only the window.
                else:
                    raw_output = self.predict_function(signal)

                # Convert elapsed time from seconds into milliseconds.
                inference_ms = (
                    time.perf_counter() - start_time
                ) * 1000.0

                # Convert the model-specific result into our common format.
                prediction = self._normalize_output(
                    raw_output,
                    inference_ms,
                )

                logger.debug(
                    "Blink prediction: %s | score=%s | inference=%.3f ms",
                    prediction.label,
                    prediction.score,
                    prediction.inference_ms,
                )

                return prediction

        except ModelContractError:
            # Preserve validation and output errors without hiding them.
            raise

        except Exception as exc:
            # Convert framework-specific errors into a consistent
            # exception that app.py can handle.
            raise ModelInferenceError(
                f"Blink model inference failed: {exc}"
            ) from exc

    # -----------------------------------------------------------------
    # MODEL CLEANUP
    # -----------------------------------------------------------------

    def close(self):
        """
        Release model resources when the application shuts down.

        Why do we need this?
        --------------------
        Some models allocate resources such as:
            - GPU memory.
            - File handles.
            - Background worker resources.
            - Framework-specific execution sessions.

        If the model exposes a close() method, we call it.

        Not every ML framework requires explicit cleanup, so this method
        checks whether close() exists before calling it.

        After cleanup, the adapter is marked as unloaded. It must be
        loaded again before another prediction can be made.
        """

        if not self.loaded:
            return

        try:
            # Check whether the loaded model provides its own cleanup method.
            model_close = getattr(self.model, "close", None)

            if callable(model_close):
                model_close()

            logger.info("Blink model resources released.")

        except Exception as exc:
            logger.exception(
                "An error occurred while closing the blink model: %s",
                exc,
            )

        finally:
            # Clear references and reset the adapter's state.
            self.model = None
            self.predict_function = None
            self.module = None
            self.loaded = False
