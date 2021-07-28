#!/usr/bin/env python3
#
# Copyright (c) 2021 LunarG, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import sys
from base_generator import write
from dx12_base_generator import Dx12BaseGenerator


class Dx12ConvertToTextsHeaderGenerator(Dx12BaseGenerator):
    """Generates C++ functions responsible for Convert to texts."""

    def __init__(
        self,
        source_dict,
        dx12_prefix_strings,
        err_file=sys.stderr,
        warn_file=sys.stderr,
        diag_file=sys.stdout
    ):
        Dx12BaseGenerator.__init__(
            self, source_dict, dx12_prefix_strings, err_file, warn_file,
            diag_file
        )

    def beginFile(self, gen_opts):
        """Methond override."""
        Dx12BaseGenerator.beginFile(self, gen_opts)

        self.write_include()

        write('GFXRECON_BEGIN_NAMESPACE(gfxrecon)', file=self.outFile)
        write('GFXRECON_BEGIN_NAMESPACE(decode)', file=self.outFile)
        self.newline()

    def generate_feature(self):
        """Methond override."""
        Dx12BaseGenerator.generate_feature(self)
        self.colloect_iid_list = list()

        self.write_enum_covert_to_text()
        self.write_iid_covert_to_text()

    def write_include(self):
        code = ''
        header_dict = self.source_dict['header_dict']
        for k, v in header_dict.items():
            code += '#include <{}>\n'.format(k)

        code += '#include "util/defines.h"\n'
        write(code, file=self.outFile)

    def write_enum_covert_to_text(self):
        enum_dict = self.source_dict['enum_dict']
        for k, v in enum_dict.items():
            code = 'static const std::string ConverttoText(const {} value)\n'\
                   '{{\n'\
                   '    switch(value)\n'\
                   '    {{\n'.format(k)

            value_set = set()
            for value in v['values']:
                if (
                    (type(value['value']) == int) or
                    (('+ 1') in value['value'])
                ) and (not value['value'] in value_set):
                    code += '        case({0}):\n'\
                            '            return "{0}";\n'.format(value['name'])
                    value_set.add(value['value'])

            code += '        default:\n'\
                    '            {{\n'\
                    '                std::string code = "Invalid {}(";\n'\
                    '                code.append(std::to_string(value));\n'\
                    '                code.append(")");\n'\
                    '                return code;\n'\
                    '            }}\n'\
                    '    }}\n'\
                    '}}\n'.format(k)
            write(code, file=self.outFile)

    def write_iid_covert_to_text(self):
        self.colloect_iid_list = list()
        header_dict = self.source_dict['header_dict']
        for k, v in header_dict.items():
            if hasattr(v, 'variables'):
                for m in v.variables:
                    if 'DEFINE_GUID' in m['type']:
                        index = m['type'].find(',')
                        self.colloect_iid_list.append(
                            m['type'][len('DEFINE_GUID ( '):index]
                        )

        code = 'static std::string ConverttoText(REFIID value)\n'\
               '{\n'\

        for iid in self.colloect_iid_list:
            code += '    if(value == {0})\n'\
                    '    {{\n'\
                    '        return "{0}";\n'\
                    '    }}\n'.format(iid)

        code += '    return "Invalid IID";\n'\
                '}\n'
        write(code, file=self.outFile)

    def endFile(self):
        """Methond override."""
        self.newline()
        write('GFXRECON_END_NAMESPACE(decode)', file=self.outFile)
        write('GFXRECON_END_NAMESPACE(gfxrecon)', file=self.outFile)

        # Finish processing in superclass
        Dx12BaseGenerator.endFile(self)