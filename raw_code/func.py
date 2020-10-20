
for i in range(x[0].shape[0]):
    _x= x.copy()
    _x[1, i, ...] *= 0.
    prob[i] = rgb_adv_model.session.run(fetches=rgb_adv_model._softmax,
                                      feed_dict={rgb_adv_model._inputs: np.expand_dims(_x[1], axis=0),
                                                 rgb_adv_model._labels: label[0][np.newaxis],
                                                 rgb_adv_model._labels_coeff: label_coeff, adv_flag: 0.0})[0][label[0]]


for i in range(x[0].shape[0]):
    _x= x.copy()
    _x[1, i, ...] *= 0.
    prob[i] = rgb_adv_model.session.run(fetches=rgb_adv_model._softmax,
                                      feed_dict={rgb_adv_model._inputs: np.expand_dims(_x[1], axis=0), adv_flag: 0.0})[0].argnax()

input_grad =tf.gradients(rgb_adv_model._loss,rgb_adv_model._inputs)
_input_grad = rgb_adv_model.session.run(fetches=input_grad,feed_dict={rgb_adv_model._inputs: x,
                                                                     rgb_adv_model._labels: label,
                                                                     rgb_adv_model._labels_coeff: label_coeff, adv_flag: 0.0} )



in_grad_norm =np.linalg.norm(_input_grad[0].reshape([79,-1]),ord=1,axis=1)


import matplotlib.pyplot as plt
x=rgb_sample
x+=rgb_adversarial.perturbed

prob=[]
for i in range(x[0].shape[0]+1):
    _x= x.copy()
    _x[0, i:, ...] *= 0.
    prob.append(rgb_adv_model.session.run(fetches=rgb_adv_model._softmax,
                                      feed_dict={rgb_adv_model._inputs: np.expand_dims(_x[0], axis=0), adv_flag: 0.0})[0])

prob = np.array(prob)
ljprob=prob[:,kinetics_classes.index('long jump')]
jtprob=prob[:,kinetics_classes.index('javelin throw')]
tjprob=prob[:,kinetics_classes.index('triple jump')]
bdprob=prob[:,kinetics_classes.index('belly dancing')]


plt.plot(tjprob, label='triple jump')
plt.plot(ljprob, label='long jump')
plt.plot(jtprob, label='javelin throw')
plt.plot(bdprob, label='belly dancing')

plt.legend(loc='upper left')

plt.savefig('out1.png')



import matplotlib.pyplot as plt
x=rgb_sample
# x+=rgb_adversarial.perturbed

prob=[]
for i in range(x[0].shape[0]):
    _x= x.copy()
    _x[0, i, ...] += rgb_adversarial.perturbed.squeeze()
    prob.append(rgb_adv_model.session.run(fetches=rgb_adv_model._softmax,
                                      feed_dict={rgb_adv_model._inputs: np.expand_dims(_x[0], axis=0), adv_flag: 0.0})[0])

prob = np.array(prob)
ljprob=prob[:,kinetics_classes.index('long jump')]
jtprob=prob[:,kinetics_classes.index('javelin throw')]
tjprob=prob[:,kinetics_classes.index('triple jump')]
# bdprob=prob[:,kinetics_classes.index('belly dancing')]


plt.plot(tjprob, label='triple jump')
plt.plot(ljprob, label='long jump')
plt.plot(jtprob, label='javelin throw')
# plt.plot(bdprob, label='belly dancing')

plt.legend(loc='upper left')

plt.savefig('out1.png')