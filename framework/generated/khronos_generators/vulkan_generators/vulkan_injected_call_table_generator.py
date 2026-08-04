#!/usr/bin/python3 -i
#
# Copyright (c) 2026 LunarG, Inc.
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
from vulkan_base_generator import VulkanBaseGenerator, VulkanBaseGeneratorOptions, write


class VulkanInjectedCallTableGeneratorOptions(VulkanBaseGeneratorOptions):
    """Options for generating the base class for VulkanInjectedCallTable."""

    def __init__(
        self,
        blacklists=None,  # Path to JSON file listing apicalls and structs to ignore.
        platform_types=None,  # Path to JSON file listing platform (WIN32, X11, etc.) defined types.
        filename=None,
        directory='.',
        prefix_text='',
        protect_file=False,
        protect_feature=True,
        extra_headers=[]
    ):
        VulkanBaseGeneratorOptions.__init__(
            self,
            blacklists,
            platform_types,
            filename,
            directory,
            prefix_text,
            protect_file,
            protect_feature,
            extra_headers=extra_headers
        )

        self.begin_end_file_data.specific_headers.extend((
            'generated/generated_vulkan_dispatch_table.h',
            'util/callbacks.h',
            'util/defines.h',
            'util/logging.h',
        ))

        self.begin_end_file_data.namespaces.extend(('gfxrecon', 'graphics'))


class VulkanInjectedCallTableGenerator(VulkanBaseGenerator):
    """VulkanInjectedCallTableGenerator - subclass of VulkanBaseGenerator.
    Generates VulkanInjectedDeviceCallsTableBase, which holds one member
    function per device command. Each member asserts the calling thread is
    inside an injected-commands scope and then forwards to the real dispatch
    table held by the base.
    """

    # Commands with hand-written members in the derived VulkanInjectedCallTable.
    MANUALLY_OVERRIDDEN_COMMANDS = ['vkBeginCommandBuffer']

    def __init__(
        self, err_file=sys.stderr, warn_file=sys.stderr, diag_file=sys.stdout
    ):
        VulkanBaseGenerator.__init__(
            self,
            err_file=err_file,
            warn_file=warn_file,
            diag_file=diag_file
        )

    def is_instance_command(self, command, first_value):
        """Matches the classification in VulkanDispatchTableGenerator."""
        return ((command in ['vkSetDebugUtilsObjectNameEXT', 'vkSetDebugUtilsObjectTagEXT']) or
                (first_value.base_type in ['VkInstance', 'VkPhysicalDevice']))

    def make_param_decl(self, value):
        """Full parameter declaration; array types keep the [] after the parameter name."""
        if value.is_array and not value.is_dynamic:
            return '{} {}[{}]'.format(value.full_type, value.name, value.array_capacity)
        else:
            return '{} {}'.format(value.full_type, value.name)

    def get_device_commands(self):
        """Collect the device commands, in dispatch-table order."""
        device_cmds = []
        for name in self.all_cmd_params:
            if self.is_core_create_command(name):
                continue

            return_type, proto, values = self.all_cmd_params[name]

            if not (values and values[0] and self.is_handle(values[0].base_type)):
                continue

            if self.is_instance_command(name, values[0]):
                continue

            device_cmds.append(name)
        return device_cmds

    def endFile(self):
        """Method override."""
        device_cmds = self.get_device_commands()

        self.newline()
        write('// One member per device command: asserts the calling thread is inside an', file=self.outFile)
        write('// injected-commands scope, then forwards to the real dispatch table.', file=self.outFile)
        self.newline()
        write('class VulkanInjectedDeviceCallsTableBase', file=self.outFile)
        write('{', file=self.outFile)
        write('  public:', file=self.outFile)
        write('    explicit VulkanInjectedDeviceCallsTableBase(const VulkanDeviceTable* raw_table) :', file=self.outFile)
        write('        raw_table_(raw_table)', file=self.outFile)
        write('    {', file=self.outFile)
        write('        GFXRECON_ASSERT(raw_table_ != nullptr);', file=self.outFile)
        write('    }', file=self.outFile)
        self.newline()
        write('    // clang-format off', file=self.outFile)

        for name in device_cmds:
            if name in self.MANUALLY_OVERRIDDEN_COMMANDS:
                write(
                    '    // {} is hand-written in the derived table.'.format(name[2:]),
                    file=self.outFile
                )
                continue

            return_type, proto, values = self.all_cmd_params[name]
            params = ', '.join([self.make_param_decl(value) for value in values])
            args = ', '.join([value.name for value in values])

            write(
                '    {} {}({}) const {{ GFXRECON_ASSERT(util::InsideInjectedCommands() && "{} called outside an InjectedCommandScope"); return raw_table_->{}({}); }}'
                .format(return_type, name[2:], params, name, name[2:], args),
                file=self.outFile
            )

        write('    // clang-format on', file=self.outFile)
        self.newline()
        write('  protected:', file=self.outFile)
        write('    const VulkanDeviceTable* raw_table_;', file=self.outFile)
        write('};', file=self.outFile)

        # Finish processing in superclass
        VulkanBaseGenerator.endFile(self)
