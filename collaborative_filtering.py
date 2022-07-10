import statistics
import json
from math import sqrt

AVERAGE_UNCOMPUTABLE = (-1000)
PEARSON_UNCOMPUTABLE_NO_COMMON = (-1001)
PEARSON_UNCOMPUTABLE_ZERO_VARIANCE = (-1002)
COSINE_UNCOMPUTABLE_NO_COMMON = (-1001)
COSINE_UNCOMPUTABLE_ZERO_VARIANCE = (-1002)
SIMILARITY_THRESHOLD = 0
MAX_RATING = 5
MIN_RATING = 1
NO_PREDICTION = -1003
MEAN_ABSOLUTE_ERROR_UNCOMPUTABLE = -1004
ROOT_MEAN_SQUARE_ERROR_UNCOMPUTABLE = -1005

class CollaborativeFiltering:

    def __init__(self, classification, use_emotion_analysis):

        if classification not in ['ekman', 'plutchik', 'poms']:
            raise ValueError('Unknown emotion classification: {}'.format(
                classification))

        self.classification = classification
        self.use_emotion_analysis = use_emotion_analysis
        self.class_dimensions = self._get_dimensions()
        self.class_weights = self._get_weights()

    def _get_dimensions(self):
        # define dimensions used in the overall process
        if self.classification == 'ekman':
            return ["Overall", "Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]
        elif self.classification == 'plutchik':
            return ["Overall", "Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust", "Anticipation"]
        elif self.classification == 'poms':
            return ["Overall", "Anger", "Depression", "Fatigue", "Vigour", "Tension", "Confusion"]
    def _get_weights(self):
        # define weights for the dimensions
        if self.classification == 'ekman':
            return {"Overall": 0.12, "Anger": 0.1, "Disgust": 0.05, "Fear": 0.05, "Joy": 0.1, "Sadness": 0.05, "Surprise": 0.05}
        elif self.classification == 'plutchik':
            return {"Overall": 0.12, "Anger": 0.05, "Disgust": 0.05, "Fear": 0.05, "Joy": 0.05, "Sadness": 0.05, "Surprise": 0.05, "Trust": 0.05, "Anticipation": 0.05}
        elif self.classification == 'poms':
            return {"Overall": 0.12, "Anger": 0.1, "Depression": 0.05, "Fatigue": 0.05, "Vigour": 0.05, "Tension": 0.1, "Confusion": 0.05}

    def avgRating(self, ratingDict):
        if (len(ratingDict) == 0):
            return AVERAGE_UNCOMPUTABLE
        return statistics.mean(ratingDict[k] for k in ratingDict)

    def avgRatingMult(self, ratingDict):
        if (len(ratingDict) == 0):
            return AVERAGE_UNCOMPUTABLE
        result = {}
        for dimension in self.class_dimensions:
            dsum = 0
            dcount = 0
            for r in ratingDict.values():
                dsum += r[dimension]
                dcount += 1
            result[dimension] = dsum / dcount
        return result

    def pearsonSim(self, x, avg_x, y, avg_y):
        if ((avg_x == AVERAGE_UNCOMPUTABLE) or (avg_y == AVERAGE_UNCOMPUTABLE)):
            return PEARSON_UNCOMPUTABLE_NO_COMMON
        nominator = 0
        denominatorX = 0
        denominatorY = 0
        count = 0
        for key in x:
            if key in y:
                xdiff = x[key] - avg_x
                ydiff = y[key] - avg_y
                nominator += xdiff * ydiff
                denominatorX += xdiff * xdiff
                denominatorY += ydiff * ydiff
                count += 1

        if (count == 0):
            return PEARSON_UNCOMPUTABLE_NO_COMMON
        else:
            if (denominatorX * denominatorY == 0):
                return PEARSON_UNCOMPUTABLE_ZERO_VARIANCE
            else:
                return nominator / sqrt(denominatorX * denominatorY)

    def pearsonSimMult(self, x, avg_x, y, avg_y):
        if ((avg_x ==  AVERAGE_UNCOMPUTABLE) or (avg_y == AVERAGE_UNCOMPUTABLE)):
          return PEARSON_UNCOMPUTABLE_NO_COMMON

        similarityDimensionNominators = {}
        similarityDimensionDenominatorsX = {}
        similarityDimensionDenominatorsY = {}
        similarityDimensionCounts = {}
        for d in self.class_dimensions:
          similarityDimensionNominators[d] = 0
          similarityDimensionDenominatorsX[d] = 0
          similarityDimensionDenominatorsY[d] = 0
          similarityDimensionCounts[d] = 0
        for key in x:
            if key in y:
              for d in self.class_dimensions:
                if (d in x[key]) and (d in y[key]):
                   xdiff = x[key][d] - avg_x[d]
                   ydiff = y[key][d] - avg_y[d]
                   similarityDimensionNominators[d] += xdiff * ydiff
                   similarityDimensionDenominatorsX[d] += xdiff * xdiff
                   similarityDimensionDenominatorsY[d] += ydiff * ydiff
                   similarityDimensionCounts[d] += 1

        pearsonNominator = 0
        pearsonDenominator = 0
        for d in self.class_dimensions:
          if ((similarityDimensionCounts[d] > 0) and (similarityDimensionDenominatorsX[d] > 0) and (similarityDimensionDenominatorsY[d] > 0)):
            pearsonNominator += self.class_weights[d] * (similarityDimensionNominators[d] / sqrt(similarityDimensionDenominatorsX[d] * similarityDimensionDenominatorsY[d]))
            pearsonDenominator += self.class_weights[d]

        if (pearsonDenominator == 0):
            return PEARSON_UNCOMPUTABLE_ZERO_VARIANCE
        else:
            return pearsonNominator / pearsonDenominator

    def cosineSim(self, x, y):
        nominator = 0
        denominatorX = 0
        denominatorY = 0
        count = 0
        for key in x:
            if key in y:
                xdiff = x[key] - MIN_RATING
                ydiff = y[key] - MIN_RATING
                nominator += xdiff * ydiff
                denominatorX += xdiff * xdiff
                denominatorY += ydiff * ydiff
                count += 1

        if (count == 0):
          return COSINE_UNCOMPUTABLE_NO_COMMON
        else:
          if (denominatorX * denominatorY == 0):
            return COSINE_UNCOMPUTABLE_ZERO_VARIANCE
          else:
            return nominator / (sqrt(denominatorX) * sqrt(denominatorY))

    def cosineSimMult(self, x, y):
        similarityDimensionNominators = {}
        similarityDimensionDenominatorsX = {}
        similarityDimensionDenominatorsY = {}
        similarityDimensionCounts = {}
        for d in self.class_dimensions:
            similarityDimensionNominators[d] = 0
            similarityDimensionDenominatorsX[d] = 0
            similarityDimensionDenominatorsY[d] = 0
            similarityDimensionCounts[d] = 0
        for key in x:
            if key in y:
                for d in self.class_dimensions:
                    if (d in x[key]) and (d in y[key]):
                        xdiff = x[key][d] - MIN_RATING
                        ydiff = y[key][d] - MIN_RATING
                        similarityDimensionNominators[d] += xdiff * ydiff * self.class_weights[d]
                        similarityDimensionDenominatorsX[d] += xdiff * xdiff * self.class_weights[d]
                        similarityDimensionDenominatorsY[d] += ydiff * ydiff * self.class_weights[d]
                        similarityDimensionCounts[d] += 1

        cosineNominator = 0
        cosineDenominatorX = 0
        cosineDenominatorY = 0
        for d in self.class_dimensions:
          if ((similarityDimensionCounts[d] > 0) and (similarityDimensionDenominatorsX[d] > 0) and (similarityDimensionDenominatorsY[d] > 0)):
            cosineNominator += similarityDimensionNominators[d]
            cosineDenominatorX += similarityDimensionDenominatorsX[d]
            cosineDenominatorY += similarityDimensionDenominatorsY[d]
        if (cosineDenominatorX == 0) or (cosineDenominatorY == 0):
            return COSINE_UNCOMPUTABLE_ZERO_VARIANCE
        else:
            return cosineNominator / (sqrt(cosineDenominatorX * cosineDenominatorY))

    def pearsonCorrelation(self, ratings):

        avgUserRatings = []
        for i, author in enumerate(ratings):
            if not self.use_emotion_analysis:
                avgUserRatings.insert(i, self.avgRating(ratings[author]))
            else:
                avgUserRatings.insert(i, self.avgRatingMult(ratings[author]))

        sim = {}
        for i, a in enumerate(ratings):
            for j, b in enumerate(ratings):
                if a in sim:
                    if self.use_emotion_analysis:
                        sim[a][b] = self.pearsonSimMult(ratings[a], avgUserRatings[i], ratings[b], avgUserRatings[j])
                    else:
                        sim[a][b] = self.pearsonSim(ratings[a], avgUserRatings[i], ratings[b], avgUserRatings[j])
                else:
                    sim[a] = {}
                    if self.use_emotion_analysis:
                        sim[a][b] = self.pearsonSimMult(ratings[a], avgUserRatings[i], ratings[b], avgUserRatings[j])
                    else:
                        sim[a][b] = self.pearsonSim(ratings[a], avgUserRatings[i], ratings[b], avgUserRatings[j])
        return sim

    def cosineSimilarity(self, ratings):
        sim = {}
        for i, a in enumerate(ratings):
            for j, b in enumerate(ratings):
                if a in sim:
                    if self.use_emotion_analysis:
                        sim[a][b] = self.cosineSimMult(ratings[a], ratings[b])
                    else:
                        sim[a][b] = self.cosineSim(ratings[a], ratings[b])
                else:
                    sim[a] = {}
                    if self.use_emotion_analysis:
                        sim[a][b] = self.cosineSimMult(ratings[a], ratings[b])
                    else:
                        sim[a][b] = self.cosineSim(ratings[a], ratings[b])
        return sim

    def predictRatingPearson(self, ratings, sim, userToPredict, itemToPredict):

        count = 0
        numerator = 0
        denominator = 0

        numUsers = len(ratings)
        avgUserRatings = []
        for i, author in enumerate(ratings):
            avgUserRatings.insert(i, self.avgRating(ratings[author]))

        for i, user in enumerate(ratings):
            if user in sim[userToPredict]:
                if (itemToPredict in ratings[user]) and (sim[userToPredict][user] > 0):
                    numerator += (ratings[user][itemToPredict] - avgUserRatings[i]) * sim[userToPredict][user]
                    denominator += abs(sim[userToPredict][user])

        if denominator > 0:
            retval = (self.avgRating(ratings[userToPredict])) + (numerator / denominator)
            if retval > MAX_RATING:
                retval = MAX_RATING
            elif retval < MIN_RATING:
                retval = MIN_RATING
            return retval
        else:
            return NO_PREDICTION

    def predictRatingCosine(self, ratings, sim, userToPredict, itemToPredict):

        count = 0
        numerator = 0
        denominator = 0

        for i, user in enumerate(ratings):
            if user in sim[userToPredict]:
                if (itemToPredict in ratings[user]) and (sim[userToPredict][user] > 0.5):
                    numerator += ratings[user][itemToPredict] * sim[userToPredict][user]
                    denominator += sim[userToPredict][user]

        if denominator > 0:
            retval = (numerator / denominator)
            if retval > MAX_RATING:
                retval = MAX_RATING
            elif retval < MIN_RATING:
                retval = MIN_RATING
            return retval
        else:
            return NO_PREDICTION

    # Mean Absolute Error (MAE)
    def mae(self, true_value, prediction):
        sum = 0
        count = 0
        for i, y_dict in enumerate(prediction):
            for key in y_dict:
                if prediction[i][key] > 0:
                    sum = sum + abs(prediction[i][key]-true_value[i][key])
                    count += 1

        if count != 0:
            return sum / count
        else:
            return MEAN_ABSOLUTE_ERROR_UNCOMPUTABLE

    # Root-Mean-Square Error (RMS Error)
    def rmse(self, true_value, prediction):
        sum = 0
        count = 0
        for i, y_dict in enumerate(prediction):
            for key in y_dict:
                if prediction[i][key] > 0:
                    sum = sum + (prediction[i][key] - true_value[i][key])**2
                    count += 1

        if count != 0:
            return sqrt(sum / count)
        else:
            return ROOT_MEAN_SQUARE_ERROR_UNCOMPUTABLE
