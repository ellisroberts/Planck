try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO
try:
    unicode
except NameError:
    unicode = str
import posixpath
import os

from django.core.exceptions import ValidationError
from django.forms.fields import MultiValueField, FileField, URLField, CharField
from django.forms.utils import ErrorList
from django.core.validators import EMPTY_VALUES
from django.core.files.uploadedfile import UploadedFile, InMemoryUploadedFile
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

import requests

from .widgets import MutuallyExclusiveRadioWidget


__all__ = ['MutuallyExclusiveValueField', 'FileOrURLField']


class MutuallyExclusiveValueField(MultiValueField):
    too_many_values_error = 'Exactly One field is required, no more'
    empty_values = EMPTY_VALUES
    labels = ()

    def __init__(self, fields=(), labels=(), *args, **kwargs):
        if 'widget' not in kwargs:
            kwargs['widget'] = MutuallyExclusiveRadioWidget(widgets=[
                field.widget for field in fields], labels=labels)
        
        ##update the name field for the widgets
        for index in range(len(fields)):
            kwargs['widget'].widgets[index].attrs['name'] = fields[index].label

        super(MutuallyExclusiveValueField, self).__init__(
            fields, *args, **kwargs)

    def clean(self, value):
        """
        Validates every value in the given list. A value is validated against
        the corresponding Field in self.fields.
        Only allows for exactly 1 valid value to be submitted, this is what
        gets returned by compress.
        example to use directy (instead of using FileOrURLField):
            MutuallyExclusiveValueField(
                fields=(forms.TypedChoiceField(choices=[(1,1), (2,2)], coerce=int),
                        forms.IntegerField()))
        """
        clean_data = []
        errors = ErrorList()
        if not value or isinstance(value, (list, tuple)):
            if not value or not [
                    v for v in value if v not in self.empty_values]:
                if self.required:
                    raise ValidationError(
                        self.error_messages['required'], code='required')
                else:
                    return self.compress([])
        else:
            raise ValidationError(
                self.error_messages['invalid'], code='invalid')
        for i, field in enumerate(self.fields):
            try:
                field_value = value[i]
            except IndexError:
                field_value = None
            try:
                clean_data.append(field.clean(field_value))
            except ValidationError as e:
                # Collect all validation errors in a single list, which we'll
                # raise at the end of clean(), rather than raising a single
                # exception for the first error we encounter.
                errors.extend(e.messages)
        if errors:
            raise ValidationError(errors)

        out = self.compress(clean_data)
        self.validate(out)
        self.run_validators(out)
        return out

    def compress(self, data_list):
        """
        Returns a single value for the given list of values. The values can be
        assumed to be valid.
        For example, if this MultiValueField was instantiated with
        fields=(DateField(), TimeField()), this might return a datetime
        object created by combining the date and time in data_list.
        """

        non_empty_list = [d for d in data_list if d not in self.empty_values]

        if len(non_empty_list) == 0 and not self.required:
            return None
        elif len(non_empty_list) > 1:
            raise ValidationError(self.too_many_values_error)

        return non_empty_list[0]

