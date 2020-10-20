# tv =tf.expand_dims(tf.get_default_graph().get_tensor_by_name('inception_i3d/Conv3d_2b_1x1/Relu:0'),axis=0)
vname = 'RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/beta:0'
vname = 'RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/beta:0'
vname = 'RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/conv_3d/w:0'
end_points = k_i3d.end_points.values()
predictions = {}
# for t in end_points:
#     # predictions[t.name]=tf.expand_dims(tf.get_default_graph().get_tensor_by_name(t.name),axis=0)
#     predictions[t.name]=tf.get_default_graph().get_tensor_by_name(t.name)
#
# tv = tf.trainable_variables()
# for t in tv:
#     # predictions[t.name]=tf.expand_dims(tf.get_default_graph().get_tensor_by_name(t.name),axis=0)
#     predictions[t.name]=tf.get_default_graph().get_tensor_by_name(t.name)
#
# predictions['input']=rgb_sample #tf.expand_dims(rgb_sample,axis=0)
# predictions['inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add_1:0'] = tf.get_default_graph().get_tensor_by_name('inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add_1:0')
predictions = {'prob': softmax}
# predictions = {
# 'tv': tv
# 'eps': eps_rgb,
# 'vid':inputs,
# 'top_1': tf.argmax(softmax, -1),
# 'prob': tf.nn.softmax(softmax)
# }
if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions)  # ,evaluation_hooks=[eval_summary_hook])

    # var =tf.get_default_graph().get_tensor_by_name('inception_i3d/Conv3d_2b_1x1/Relu:0')
    var = var[:, 40, 40, 40, 40]

    var2 = tf.get_default_graph().get_tensor_by_name('RGB/inception_i3d/Conv3d_2b_1x1/conv_3d/w:0')
    var2 = var2[0, 0, 0, 0, 5]
    # return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # tf.get_default_graph().get_tensor_by_name('RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/beta:0')
    hook = tf.train.LoggingTensorHook({"var is:": var, "var 2 is:": var2},
                                      every_n_iter=1)
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions,
        evaluation_hooks=[hook])  # ,evaluation_hooks=[eval_summary_hook])

