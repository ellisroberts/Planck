from django.forms.widgets import Input, MultiWidget, TextInput
from django.core.validators import EMPTY_VALUES
from django.utils.safestring import mark_safe
from django.core.files.uploadedfile import UploadedFile
import pdb

try:
    unicode
except NameError:
    unicode = str

__all__ = ['RadioInput', 'URLInput', 'MutuallyExclusiveRadioWidget',
           'FileOrURLWidget']


class RadioInput(Input):
    input_type = 'radio'


class MutuallyExclusiveRadioWidget(MultiWidget):

    def __init__(self, labels=(), *args, **kwargs):
        self.labels = labels
        super(MutuallyExclusiveRadioWidget, self).__init__(
            *args, **kwargs)

    def render(self, name, value, attrs=None):
        if self.is_localized:
            for widget in self.widgets:
                widget.is_localized = self.is_localized
        # value is a list of values, each corresponding to a widget
        # in self.widgets.
        if not isinstance(value, list):
            value = self.decompress(value)
        output = []
        final_attrs = self.build_attrs(attrs)
        id_ = final_attrs.get('id', None)
        nonempty_widget = 0
        for i, widget in enumerate(self.widgets):
            try:
                widget_value = value[i]
            except IndexError:
                widget_value = None
            else:
                if widget_value not in EMPTY_VALUES:
                    nonempty_widget = i
            if id_:
                final_attrs = dict(final_attrs, id='%s_%s' % (id_, i))
            output.append(widget.render(
                widget.attrs['name'], widget_value, final_attrs))
        outputStr = mark_safe(unicode(
            self.format_output(nonempty_widget, name, output)))
        return outputStr

    def format_output(self, nonempty_widget, name, rendered_widgets):
        radio_values = [str(index) for index in range(len(rendered_widgets))]
        radio_widgets = [RadioInput(attrs={"value":str(radio_values[index])}).render(
            name + '_radio', '', {}) for index in range(len(radio_values))]
        input_labels = ["""<label for="ModelInfo"> {0} </label>""".format(x) for x in self.labels]

        if nonempty_widget is not None:
            value = radio_values[nonempty_widget]
            radio_widgets[nonempty_widget] = RadioInput(attrs={'value':str(value)}).render(
                name + '_radio', '', {'checked': ''}) 
        tpl = """
<span id="{name}_container" class="mutually-exclusive-widget"
    style="display:inline-block">
    {widgets}
</span>"""

        newTpl = tpl.format(name=name, widgets='<br>'.join(
            '<span>{0}</span>'.format(x + y + z)
            for x, y, z in zip(radio_widgets, rendered_widgets, input_labels)))

        return newTpl

    def decompress(self, value):
        """
        If initialized with single compressed value we don't know what to do.
        just so it doesn't 'splode, return empty list of right length
        """
        return [''] * len(self.widgets)

    class Media:
        js = ('clientjs/mutually_exclusive_widget.js',)
