import tensorflow as tf
import numpy as np
import cv2

# =========================
# GRAD-CAM HEATMAP
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block16_concat"):

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array], training=False)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    return heatmap


# =========================
# OVERLAY FUNCTION
# =========================
def get_gradcam_overlay(img, model):

    img_input = np.expand_dims(img, axis=0)

    heatmap = make_gradcam_heatmap(img_input, model)

    heatmap = cv2.resize(heatmap, (224, 224))

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    img_uint8 = np.uint8(img * 255)

    overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)

    return overlay