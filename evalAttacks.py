import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F

from cleverhans.compat import flags
from cleverhans.train import train
from cleverhans.dataset import MNIST
from cleverhans.model import CallableModelWrapper
# from cleverhans.loss import CrossEntropy
from cleverhans.utils import AccuracyReport
from cleverhans.utils_tf import model_eval
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

# change to completely pytorch
from cleverhans.future.torch.attacks.fast_gradient_method import fast_gradient_method

#advertorch attacks
from advertorch.attacks import CarliniWagnerL2Attack

def Ensembler(preds):
    finalPred = np.zeros(len(preds[0]))
    for i in range(len(preds[0])):
      scoreList = np.zeros(10)
      for pred in preds:
        scoreList[pred[i]] += 1
      finalPred[i] = np.argmax(scoreList)
    return finalPred

def evalClean(model1=None, model2=None, model3=None, model4=None, test_loader=None, report=None, singleModel=0):
    if singleModel:
      print("Evaluating single model results on clean data")
    else:
      print("Evaluating the ensembled method on clean data")
    total = 0
    correct = 0
    with torch.no_grad():
      model1.eval()
      if not singleModel:
        model2.eval()
        model3.eval()
        # only when 4 models
        # model4.eval()
      for xs, ys in test_loader:
        xs, ys = Variable(xs), Variable(ys)
        if torch.cuda.is_available():
          xs, ys = xs.cuda(), ys.cuda()
        preds1 = model1(xs)
        preds_np1 = preds1.cpu().detach().numpy()
        if not singleModel:
          preds2 = model2(xs)
          preds_np2 = preds2.cpu().detach().numpy()
          preds3 = model3(xs)
          preds_np3 = preds3.cpu().detach().numpy()
          # only when 4 models
          # preds4 = model4(xs)
          # preds_np4 = preds4.cpu().detach().numpy()
          #preds for 3 and 4
          preds = [np.argmax(preds_np1, axis=1), np.argmax(preds_np2, axis=1), np.argmax(preds_np3, axis=1)]
          # preds = [np.argmax(preds_np1, axis=1), np.argmax(preds_np2, axis=1), np.argmax(preds_np3, axis=1), np.argmax(preds_np4, axis=1)]
          finalPred = Ensembler(preds)
        else:
          finalPred = np.argmax(preds_np1, axis=1)
        correct += (finalPred == ys.cpu().detach().numpy()).sum()
        total += len(xs)
    acc = float(correct) / total
    print('Clean accuracy: %.2f%%' % (acc * 100))

    return report
def evalFGSMEnsem(fgsm_model, models, test_loader):
    total = 0
    correct = 0
    fgsm_model.eval()
    for model in models:
        model.eval()
    for xs, ys in test_loader:
        if torch.cuda.is_available():
            xs, ys = xs.cuda(), ys.cuda()
        #pytorch fast gradient method
        xs = fast_gradient_method(fgsm_model, xs, eps=0.3, norm=np.inf, clip_min=0., clip_max=1.)
        xs, ys = Variable(xs), Variable(ys)
        preds1 = fgsm_model(xs)
        preds_np1 = preds1.cpu().detach().numpy()
        preds = [np.argmax(preds_np1, axis=1)]
        for model in models:
            predsT = model(xs)
            predsT_np = predsT.cpu().detach().numpy()
            preds.append(np.argmax(predsT_np, axis=1))
        finalPred = Ensembler(preds)
        correct += (finalPred == ys.cpu().detach().numpy()).sum()
        total += test_loader.batch_size
    acc = float(correct) / total
    print('Adv accuracy: {:.3f}％'.format(acc * 100))

def evalAdvAttack(fgsm_model=None, model2=None, model3=None, model4=None, test_loader=None, singleModel=0):
    # fgsm_model
    # fast_gradient_method(model_fn, x, eps, norm, clip_min=None, clip_max=None, y=None, targeted=False, sanity_checks=False):
    if singleModel:
      print("Evaluating single model results on adv data")
    else:
      print("Evaluating the ensembled method on adv data")
    total = 0
    correct = 0
    fgsm_model.eval()
    if not singleModel:
      model2.eval()
      model3.eval()
      # only when 4 models
      # model4.eval()

    for xs, ys in test_loader:
      if torch.cuda.is_available():
        xs, ys = xs.cuda(), ys.cuda()
      #pytorch fast gradient method
      xs = fast_gradient_method(fgsm_model, xs, eps=0.3, norm=np.inf, clip_min=0., clip_max=1.)
      # xs = fast_gradient_method(fgsm_model, xs, eps=0.1, norm=np.inf)
      xs, ys = Variable(xs), Variable(ys)
      preds1 = fgsm_model(xs)
      # loss = F.nll_loss(preds1, ys)
      # loss.backward()  # calc gradients
      # train_loss.append(loss.data.item())
      # optimizer.step()  # update gradients
      preds_np1 = preds1.cpu().detach().numpy()
      if not singleModel:
          preds2 = model2(xs)
          preds_np2 = preds2.cpu().detach().numpy()
          preds3 = model3(xs)
          preds_np3 = preds3.cpu().detach().numpy()
          # only when 4 models
          # preds4 = model4(xs)
          # preds_np4 = preds4.cpu().detach().numpy()
          #preds for 3 and 4
          preds = [np.argmax(preds_np1, axis=1), np.argmax(preds_np2, axis=1), np.argmax(preds_np3, axis=1)]
          # preds = [np.argmax(preds_np1, axis=1), np.argmax(preds_np2, axis=1), np.argmax(preds_np3, axis=1), np.argmax(preds_np4, axis=1)]
          finalPred = Ensembler(preds)
      else:
          finalPred = np.argmax(preds_np1, axis=1)
      correct += (finalPred == ys.cpu().detach().numpy()).sum()
      total += test_loader.batch_size
    acc = float(correct) / total
    print('Adv accuracy: {:.3f}％'.format(acc * 100))
      # break

def evalAdvDiffModel(fgsm_model, modelTest, test_loader):
    # fgsm_model generates the fgsm, the attack is then tested on modelTest
    # fast_gradient_method(model_fn, x, eps, norm, clip_min=None, clip_max=None, y=None, targeted=False, sanity_checks=False):
    # if singleModel:
    #   print("Evaluating single model results on adv data")
    # else:
    print("Evaluating the adv data with another model")
    total = 0
    correct = 0
    fgsm_model.eval()
    modelTest.eval()
    # if not singleModel:
    #   model2.eval()
    #   model3.eval()
    #   # only when 4 models
      # model4.eval()

    for xs, ys in test_loader:
      if torch.cuda.is_available():
        xs, ys = xs.cuda(), ys.cuda()
      #pytorch fast gradient method
      xs = fast_gradient_method(fgsm_model, xs, eps=0.3, norm=np.inf, clip_min=0., clip_max=1.)
      # xs = fast_gradient_method(fgsm_model, xs, eps=0.1, norm=np.inf)
      xs, ys = Variable(xs), Variable(ys)
      preds1 = modelTest(xs)
      preds_np1 = preds1.cpu().detach().numpy()
      finalPred = np.argmax(preds_np1, axis=1)
      correct += (finalPred == ys.cpu().detach().numpy()).sum()
      total += test_loader.batch_size
    acc = float(correct) / total
    print('Adv accuracy: {:.3f}％'.format(acc * 100))
      # break

def evalCombined(model1, model2, model3, test_loader, report):
    # This function evaluates the main model (model 1) and the ensembled data (model1/2/3) on clean and adv data

    # Evaluate single model on clean data
    evalClean(model1=model1, test_loader=test_loader, report=report, singleModel=1)
    # Evaluate ensembled model on clean data
    evalClean(model1=model1, model2=model2, model3=model3, test_loader=test_loader, report=report, singleModel=0)
    # Evaluate single model on adversarial attack
    evalAdvAttack(fgsm_model=model1, test_loader=test_loader, singleModel=1)
    # Evaluate ensembled model on adversarial attack
    evalAdvAttack(fgsm_model=model1, model2=model2, model3=model3, test_loader=test_loader, singleModel=0)


#Ensemble attack via random ensembling
def CWAttackRandomEnsem(ensemModel, modelGroup, test_loader):
    ensemModel.eval()
    for model in modelGroup:
      model.eval()

    correctEn = 0
    correctRn = 0
    total = 0

    #Define C&W Attack
    adversary = CarliniWagnerL2Attack(predict=ensemModel, num_classes=10)
    numberBatch = 0

    for xs, ys in test_loader:
      print("On Batch {}".format(numberBatch+1))
      if torch.cuda.is_available():
        xs, ys = xs.cuda(), ys.cuda()

      #Perform the attack
      xs = adversary.perturb(xs, ys)
      xs, ys = Variable(xs), Variable(ys)

      #Testing the original
      print("Running on the original Ensembled")
      preds = ensemModel(xs)
      preds_np0 = preds.cpu().detach().numpy()
      finalPred = np.argmax(preds_np0, axis=1)
      correctEn += (finalPred == ys.cpu().detach().numpy()).sum()

      #Testing the random ensemble
      print('Running on the random Ensembled')
      #add random n-1 samples into the ensemble voting system
      elim = np.random.randint(len(modelGroup))
      preds = []

      for i in range(len(modelGroup)):
        if i == elim:
          continue
        preds.append(np.argmax(modelGroup[i](xs).cpu().detach().numpy(), axis=1))
      finalPred = Ensembler(preds)

      correctRn += (finalPred == ys.cpu().detach().numpy()).sum()


      total += test_loader.batch_size
      numberBatch += 1
      if numberBatch == 2:
        break
    acc = float(correctEn) / total
    print('Adv accuracy for original ensem model: {:.3f}％'.format(acc * 100))

    acc = float(correctRn) / total
    print('Adv accuracy for tested model 1: {:.3f}％'.format(acc * 100))



#Ensemble attack
def CWAttackEnsem(ensemModel, modelGroup, test_loader):
    ensemModel.eval()

    for model in modelGroup:
      model.eval()

    correctEn = 0
    corrects = [0 for i in range(len(modelGroup))]
    correctF = 0
    total = 0

    #Define C&W Attack
    adversary = CarliniWagnerL2Attack(predict=ensemModel, num_classes=10)
    numberBatch = 0

    for xs, ys in test_loader:
      print("On Batch {}".format(numberBatch+1))
      if torch.cuda.is_available():
        xs, ys = xs.cuda(), ys.cuda()
      #Perform the attack
      xs = adversary.perturb(xs, ys)
      xs, ys = Variable(xs), Variable(ys)

      #Testing the original
      print("Running on the original Ensembled")
      preds = ensemModel(xs)
      preds_np0 = preds.cpu().detach().numpy()
      finalPred = np.argmax(preds_np0, axis=1)
      correctEn += (finalPred == ys.cpu().detach().numpy()).sum()


      preds = []
      for i in range(len(modelGroup)):
        pred = np.argmax(modelGroup[i](xs).cpu().detach().numpy(), axis=1)
        preds.append(pred)
        corrects[i] += (pred == ys.cpu().detach().numpy()).sum()

      finalPred = Ensembler(preds)
      correctF += (finalPred == ys.cpu().detach().numpy()).sum()

      total += test_loader.batch_size

      numberBatch += 1
      if numberBatch == 2:
        break
    acc = float(correctEn) / total
    print('Adv accuracy for original ensem model: {:.3f}％'.format(acc * 100))


    for i in range(len(corrects)):
      acc = float(corrects[i])/total
      print('Adv accuracy for test model {}: {:.3f}%'.format(i, acc * 100))

    acc = float(correctF) / total
    print('Adv accuracy for Ensemble Voting: {:.3f}％'.format(acc * 100))

# def CWAttack(fgsm_model, test_model1, test_model2, ensemmodel, test_loader):
#     fgsm_model.eval()
#     correctOriginal = 0
#     totalOriginal = 0
#     correctAnother1 = 0
#     totalAnother1 = 0
#     correctAnother2 = 0
#     totalAnother2 = 0
#     correctEnsem = 0
#     totalEnsem = 0
#
#     #Define C&W Attack
#     adversary = CarliniWagnerL2Attack(predict=fgsm_model, num_classes=10)
#     numberBatch = 0
#
#     for xs, ys in test_loader:
#       print("Testing only 1 batch")
#       if torch.cuda.is_available():
#         xs, ys = xs.cuda(), ys.cuda()
#       #Perform the attack
#       xs = adversary.perturb(xs, ys)
#       xs, ys = Variable(xs), Variable(ys)
#
#       #Testing the original
#       print("Running on the original")
#       preds = fgsm_model(xs)
#       preds_np1 = preds.cpu().detach().numpy()
#       finalPred = np.argmax(preds_np1, axis=1)
#       correctOriginal += (finalPred == ys.cpu().detach().numpy()).sum()
#       totalOriginal += test_loader.batch_size
#
#       print("Running on test model1")
#       preds = test_model1(xs)
#       preds_np2 = preds.cpu().detach().numpy()
#       finalPred = np.argmax(preds_np2, axis=1)
#       correctAnother1 += (finalPred == ys.cpu().detach().numpy()).sum()
#       totalAnother1 += test_loader.batch_size
#
#       print("Running on test model2")
#       preds = test_model2(xs)
#       preds_np3 = preds.cpu().detach().numpy()
#       finalPred = np.argmax(preds_np3, axis=1)
#       correctAnother2 += (finalPred == ys.cpu().detach().numpy()).sum()
#       totalAnother2 += test_loader.batch_size
#
#       print("Running ensemble")
#       preds = [np.argmax(preds_np1, axis=1), np.argmax(preds_np2, axis=1), np.argmax(preds_np3, axis=1)]
#       finalPred = Ensembler(preds)
#       correctEnsem += (finalPred == ys.cpu().detach().numpy()).sum()
#       totalEnsem += test_loader.batch_size
#
#       numberBatch += 1
#       if numberBatch == 2:
#         break
#     acc = float(correctOriginal) / totalOriginal
#     print('Adv accuracy for original model: {:.3f}％'.format(acc * 100))
#
#     acc = float(correctAnother1) / totalAnother1
#     print('Adv accuracy for tested model 1: {:.3f}％'.format(acc * 100))
#
#     acc = float(correctAnother2) / totalAnother2
#     print('Adv accuracy for tested model 2: {:.3f}％'.format(acc * 100))
#
#     acc = float(correctEnsem) / totalEnsem
#     print('Adv accuracy for Ensemble: {:.3f}％'.format(acc * 100))
