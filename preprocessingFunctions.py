# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:36:56 2018

@author: Franklin
"""
def classToGrade(number):
	if number == 4:
		return 'A'
	elif number == 3:
		return 'B'
	elif number == 2:
		return 'C'
	elif number == 1:
		return 'D'
	else:
		return 'F'

def encode_grades_to_pass_fail(df):
	for atr in ['G3']:
		for i in range(0, df.shape[0]):
			if (10 <=df.at[i, atr] <= 20): # row i, column atr
				df.at[i, atr] = 1
			else:
				df.at[i, atr] = 0

def encode_grades_to_5_class(df): # A, B, C, D, F
	for atr in ['G3']:
		for i in range(0, df.shape[0]):
			if (16 <=df.at[i, atr] <= 20): # row i, column atr
				df.at[i, atr] = 4
			elif (14 <=df.at[i, atr] <= 15):
				df.at[i, atr] = 3
			elif (12 <=df.at[i, atr] <= 13):
				df.at[i, atr] = 2
			elif (10 <=df.at[i, atr] <= 11):
				df.at[i, atr] = 1
			else:
				df.at[i, atr] = 0