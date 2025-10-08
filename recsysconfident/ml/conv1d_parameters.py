
class Parameters:
    ftype: str
    input_size: int
    previous_channels: int
    kernel_size: int
    strides: int
    output_size: int
    channels: int


class Conv1DParameters:

    def __init__(self, input_size: int, output_size: int, min_channels: int, max_channels: int):
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.input_size = input_size
        self.output_size = output_size

    def channels(self, current_output: int):

        if (self.input_size - self.output_size) == 0:
            return self.max_channels

        a = -(self.max_channels - self.min_channels) / (self.input_size - self.output_size)
        b = self.min_channels - a * self.input_size

        channels = a * current_output + b

        return int(channels) + 1

    def calc_remain_input(self, current_input_size: int, k: int, s: int) -> dict:

        remaining_input = int(self.calc_output_size(current_input_size, k, s, 0))
        diff = remaining_input - self.output_size

        if (remaining_input - self.output_size) < k and diff > 0:
            k = k + diff
            remaining_input = int(self.calc_output_size(current_input_size, k, s, 0))

        return {
            "input_size": current_input_size,
            "kernel_size": k,
            "strides": s,
            "output_size": remaining_input,
            "channels": int(self.channels(current_input_size))
        }

    def get_configs(self, input_channels: int, kernel_size: int, down_target: int = 640, start_stride=2) -> list:

        k, s = kernel_size, start_stride
        config = self.calc_remain_input(self.input_size, k, s)
        config['previous_channels'] = input_channels
        qtd = 0
        current_input_size = None
        configs = []
        while config['output_size'] >= self.output_size:

            qtd += 1
            current_config = config
            current_input_size = config['output_size']

            if config['output_size'] > down_target:
                config = self.calc_remain_input(current_input_size, k, s)
            else:
                config = self.calc_remain_input(current_input_size, k, 1)
            config['previous_channels'] = current_config['channels']

            configs.append(current_config)

        configs[-1]['channels'] = self.max_channels

        return configs

    def calc_output_size(self, input_size: int, kernel_size: int, stride: int, padding: int = 0) -> int:

        return int((input_size - kernel_size + 2 * padding) / stride + 1)
