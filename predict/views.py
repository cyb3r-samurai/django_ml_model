from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from predict.models import PredResults
import pickle
import numpy as np
from lightgbm import LGBMRegressor
import sklearn

def predict(request):
    return render(request, 'predict.html')


def predict_chances(request):

    if request.POST.get('action') == 'post':

        # Receive data from client
        fixed_acidity = float(request.POST.get('fixed_acidity'))
        volatile_acidity = float(request.POST.get('volatile_acidity'))
        citric_acid = float(request.POST.get('citric_acid'))
        residual_sugar = float(request.POST.get('residual_sugar'))
        chlorides = float(request.POST.get('chlorides'))
        free_sulfur_dioxide = float(request.POST.get('free_sulfur_dioxide'))
        total_sulfur_dioxide = float(request.POST.get('total_sulfur_dioxide'))
        density = float(request.POST.get('density'))
        pH = float(request.POST.get('pH'))
        sulphates = float(request.POST.get('sulphates'))
        alcohol = float(request.POST.get('alcohol'))
        # Unpickle model
        model = pickle.load(open('/home/andrey/Desktop/ml3/wine/best.pkl','rb'))
        # Make prediction

        input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                chlorides, free_sulfur_dioxide,
                                total_sulfur_dioxide, density, pH,
                                sulphates, alcohol
                                ]])

        result = model.predict(input_data)

        #result = model.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides ,free_sulfur_dioxide,
        #                        total_sulfur_dioxide, density, pH, sulphates, alcohol]])

        regression = result[0]

        PredResults.objects.create(fixed_acidity=fixed_acidity, volatile_acidity=volatile_acidity, citric_acid=citric_acid,
                                   residual_sugar=residual_sugar, chlorides=chlorides, free_sulfur_dioxide=free_sulfur_dioxide,
                                   total_sulfur_dioxide=total_sulfur_dioxide, density=density, pH=pH, sulphates = sulphates,
                                   alcohol=alcohol,
                                   regression=regression)

        return JsonResponse({'result': regression, 'fixed_acidity': fixed_acidity,
                             'volatile_acidity': volatile_acidity, 'citric_acid': citric_acid, 'residual_sugar': residual_sugar,
                             'chlorides': chlorides, 'free_sulfur_dioxide': free_sulfur_dioxide,
                             'total_sulfur_dioxide': total_sulfur_dioxide, 'density': density,
                             'pH': pH, 'sulphates': sulphates, 'alcohol': alcohol},
                            safe=False)


def view_results(request):
    # Submit prediction and show all
    data = {"dataset": PredResults.objects.all()}
    return render(request, "results.html", data)
