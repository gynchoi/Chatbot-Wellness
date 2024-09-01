import openpyxl
import random
from openpyxl import Workbook, load_workbook

def wellness_autoregressive_data():
  path = "C:/Users/gynchoi/project_python/Yonsei/DL_FP/dataset"
  answer = path + "/wellness_answer.txt"
  question = path + "/wellness_question.txt"
  autoregressive = path + "/wellness_autoregressive.txt"

  f_answer = open(answer, 'r')
  f_question = open(question, 'r')
  f_autoregressive = open(autoregressive, 'w')

  answer_lines = f_answer.readlines()
  question_lines = f_question.readlines()
  
  for _, data_line in enumerate(question_lines):
    data_question = data_line.split('    ')
    for _, ans_line in enumerate(answer_lines):
      data_answer = ans_line.split('    ')
      if data_question[0] == data_answer[0]:
        f_autoregressive.write(data_question[1][:-1] + "    " + data_answer[1])
      else:
        continue

  f_answer.close()
  f_question.close()
  f_autoregressive.close()
  

def seperate_wellness_data():
  path = "../dataset"
  autoregressive = path + "/wellness_autoregressive.txt"
  train_dataset = path + "/wellness_train.txt"
  test_dataset = path + "/wellness_test.txt"
  
  f_autoregressive = open(autoregressive, 'r')
  f_train = open(train_dataset, 'w')
  f_test = open(test_dataset, 'w')

  lines = f_autoregressive.readlines()
  for _, data in enumerate(lines):
    rand_num = random.randint(0, 10)
    if rand_num < 10:
      f_train.write(data)
    else:
      f_test.write(data)

  f_autoregressive.close()
  f_train.close()
  f_test.close()
