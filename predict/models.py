from django.db import models


class PredResults(models.Model):

    fixed_acidity = models.FloatField()
    volatile_acidity = models.FloatField()
    citric_acid = models.FloatField()
    residual_sugar = models.FloatField()
    chlorides = models.FloatField()
    free_sulfur_dioxide = models.FloatField()
    total_sulfur_dioxide = models.FloatField()
    density = models.FloatField()
    pH = models.FloatField()
    sulphates = models.FloatField()
    alcohol = models.FloatField()

    regression = models.FloatField()

    def __str__(self):
        return self.regression


# Create your models here.
