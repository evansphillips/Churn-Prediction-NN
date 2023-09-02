import tensorflow as tf

def f1_score(y_true, y_pred):
    """
    Calculate F1 score using TensorFlow functions.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: F1 score.
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

def precision_m(y_true, y_pred):
    """
    Calculate precision using TensorFlow functions.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: Precision.
    """
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall_m(y_true, y_pred):
    """
    Calculate recall using TensorFlow functions.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: Recall.
    """
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    actual_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
    return recall