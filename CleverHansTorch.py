# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import numpy as np
import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import flags
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

FLAGS = flags.FLAGS

NB_EPOCHS = 2
BATCH_SIZE = 128
LEARNING_RATE = .001


warnings.warn("convert_pytorch_model_to_tf is deprecated, switch to"
              + " dedicated PyTorch support provided by CleverHans v4.")



class PytorchMnistModel(nn.Module):
  def __init__(self):
    super(PytorchMnistModel, self).__init__()
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
    # feature map size is 14*14 by pooling
    # padding=2 for same padding
    self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
    # feature map size is 7*7 by pooling
    self.fc1 = nn.Linear(64 * 7 * 7, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 64 * 7 * 7)  # reshape Variable
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=-1)

class VGGNet(nn.Module):
   def __init__(self):
        super(VGGNet, self).__init__()
        self.conv11 = nn.Conv2d(1, 64, 3)
        self.conv12 = nn.Conv2d(64, 64, 3)
        self.conv21 = nn.Conv2d(64, 128, 3)
        self.conv22 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(128 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 10)
   def forward(self, x):
       #1, 28, 28
       x = F.relu(self.conv11(x))
       #64, 26, 26
       x = F.relu(self.conv12(x))
       #64, 24, 24
       x = F.max_pool2d(x, (2,2))
       #64, 12, 12
       x = F.relu(self.conv21(x))
       #128, 10, 10
       x = F.relu(self.conv22(x))
       #128, 8, 8
       x = F.max_pool2d(x, (2,2))
       #128, 4, 4
       x = F.max_pool2d(x, (2,2))
       #128, 2, 2
       x = x.view(-1, 128 * 2 * 2)
       x = F.relu(self.fc1(x))
       x = self.fc2(x)
       return F.log_softmax(x, dim=-1)


class ensembleModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(ensembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x = x1+x2
        return F.log_softmax(x, dim=-1)

#1. Write Load and Save Checkpt
#2. Test if Pretrained Net Work
#3. Collect Adv. result for 3 nets
#4. Train them together on google colab
#5. Collect Adv. result for 3 nets combined
def train(torch_model, train_loader, test_loader,
        nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE, train_end=-1, test_end=-1, learning_rate=LEARNING_RATE):


    # Truncate the datasets so that our test run more quickly
  #   train_loader.dataset.train_data = train_loader.dataset.train_data[:train_end]
  #   test_loader.dataset.test_data = test_loader.dataset.test_data[:test_end]

    # Train our model
    optimizer = optim.Adam(torch_model.parameters(), lr=learning_rate)
    train_loss = []

    total = 0
    correct = 0
    step = 0
    breakstep = 0
    for _epoch in range(nb_epochs):
      if breakstep == 2:
          # print("break all!")
          break
      for xs, ys in train_loader:
        xs, ys = Variable(xs), Variable(ys)
        if torch.cuda.is_available():
          xs, ys = xs.cuda(), ys.cuda()
        optimizer.zero_grad()
        preds = torch_model(xs)
        loss = F.nll_loss(preds, ys)
        loss.backward()  # calc gradients
        train_loss.append(loss.data.item())
        optimizer.step()  # update gradients

        preds_np = preds.cpu().detach().numpy()
        correct += (np.argmax(preds_np, axis=1) == ys.cpu().detach().numpy()).sum()
        total += train_loader.batch_size
        step += 1
        if total % 1000 == 0:
          acc = float(correct) / total
          print('[%s] Training accuracy: %.2f%%' % (step, acc * 100))
          total = 0
          correct = 0
          breakstep += 1
          if breakstep == 2:
              # print("break!")
              break
    return step

def AttackOnModel(torch_model, test_loader, report):
    # We use tf for evaluation on adversarial data
    sess = tf.Session()
    x_op = tf.placeholder(tf.float32, shape=(None, 1, 28, 28,))

    # Convert pytorch model to a tf_model and wrap it in cleverhans
    tf_model_fn = convert_pytorch_model_to_tf(torch_model)
    cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

    # Create an FGSM attack
    fgsm_op = FastGradientMethod(cleverhans_model, sess=sess)
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x_op = fgsm_op.generate(x_op, **fgsm_params)
    adv_preds_op = tf_model_fn(adv_x_op)

    # Run an evaluation of our model against fgsm
    total = 0
    correct = 0
    for xs, ys in test_loader:
      adv_preds = sess.run(adv_preds_op, feed_dict={x_op: xs})
      correct += (np.argmax(adv_preds, axis=1) == ys.cpu().detach().numpy()).sum()
      total += test_loader.batch_size

    acc = float(correct) / total
    print('Adv accuracy: {:.3f}％'.format(acc * 100))
    report.clean_train_adv_eval = acc
    return report

def mnist_test(nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   train_end=-1, test_end=-1, learning_rate=LEARNING_RATE, torch_model=None, usePreTrain=0):
  """
  MNIST cleverhans tutorial
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :return: an AccuracyReport object
  """
      # Train a pytorch MNIST model
  if torch.cuda.is_available():
    torch_model = torch_model.cuda()

  report = AccuracyReport()
  train_loader = torch.utils.data.DataLoader(
      datasets.MNIST('data', train=True, download=True,
                     transform=transforms.ToTensor()),
      batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(
      datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
      batch_size=batch_size)
  step = 0
  if not usePreTrain:
      step = train(torch_model, train_loader, test_loader, nb_epochs, batch_size, train_end, test_end, learning_rate)
  # Evaluate on clean data
  total = 0
  correct = 0
  for xs, ys in test_loader:
    xs, ys = Variable(xs), Variable(ys)
    if torch.cuda.is_available():
      xs, ys = xs.cuda(), ys.cuda()

    preds = torch_model(xs)
    preds_np = preds.cpu().detach().numpy()

    correct += (np.argmax(preds_np, axis=1) == ys.cpu().detach().numpy()).sum()
    total += len(xs)

  acc = float(correct) / total
  report.clean_train_clean_eval = acc
  print('[%s] Clean accuracy: %.2f%%' % (step, acc * 100))
  report = AttackOnModel(torch_model, test_loader, report)

def main(_=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)
  firstModel = PytorchMnistModel()
  print('Running First Model')
  mnist_test(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate,
            torch_model=firstModel, usePreTrain=0)
  print('Running Second Model')
  secondModel = VGGNet()
  mnist_test(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate,
            torch_model=secondModel, usePreTrain=0)
  print('Running Third Model')
  thirdModel = ensembleModel(firstModel, secondModel)
  mnist_test(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate,
            torch_model=thirdModel, usePreTrain=0)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')

  tf.app.run()
