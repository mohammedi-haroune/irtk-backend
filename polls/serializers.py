from rest_framework import serializers
from collections import OrderedDict

#a document is identified by a fileid
#for all serializers that need a document reference (vector model result for example)
#we add a fileid field to the serializer

class NonNullSerializer(serializers.Serializer):
    def to_representation(self, instance):
        result = super().to_representation(instance)
        return OrderedDict([(key, result[key]) for key in result if result[key] is not None])

class DocumentSerializer(NonNullSerializer):
    fileid = serializers.CharField()
    sample = serializers.CharField(required=False)

class VectorModelResultSerializer(NonNullSerializer):
    fileid = serializers.CharField()
    inner_product = serializers.FloatField(required=False)
    dice_coef = serializers.FloatField(required=False)
    cosinus_measure = serializers.FloatField(required=False)
    jaccard_coef = serializers.FloatField(required=False)

class FindResultSerializer(NonNullSerializer):
    fileid = serializers.CharField()
    freq = serializers.FloatField(required=False)
    tf_idf = serializers.FloatField(required=False)

class UploadCorpusSerializer(NonNullSerializer):
    name = serializers.CharField()
    file = serializers.FileField()

class CorporaSerializer(NonNullSerializer):
    name = serializers.CharField()
    description = serializers.CharField()
    words = serializers.IntegerField(required=False)
    files = serializers.ListField(child=serializers.CharField(), required=False)
    categories = serializers.ListField(child=serializers.CharField(), required=False)