import time
import gym
from enum import Enum
from typing import Any, Callable, Optional

'''
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)
'''

class Color(Enum):
    GRAY       = 30
    RED        = 31
    GREEN      = 32
    YELLOW     = 33
    BLUE       = 34
    MAGENTA    = 35
    CYAN       = 36
    WHITE      = 37
    CRIMSON    = 38


def colorize(string: str, color=Color.WHITE, bold=True, highlight=False) -> str:
    attr = []
    # num = color2num[color]
    num = color.value
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class LogLevel(Enum):
    SUCCESS = Color.GREEN
    DEBUG = Color.MAGENTA
    INFO = Color.BLUE
    WARNING = Color.YELLOW
    ERROR = Color.RED

LogSymbol = {
    LogLevel.SUCCESS: "✔", # Enum key is type str
    LogLevel.DEBUG: "",
    LogLevel.INFO: "",
    LogLevel.WARNING: "⚠",
    LogLevel.ERROR: "✖"
}


def debug_msg(
        msg: str,
        level=LogLevel.DEBUG,
        color : Optional[Color] = None,
        bold=False,
        inline=False
    ):
    """
    return:
    symbol msg (same color)
    e.g.
    ✔  SUCCESS
      DEBUG
      INFO
    ⚠  WARNING
    ✖  ERROR
    """
    ''' ➤  ☞ ⚑ '''
    def colored_prompt(prompt: str) -> str:
        symbol = LogSymbol[level]
        text = symbol + ' [' + prompt + ']'
        return colorize(text, color=level.value, bold=True)

    '''prompt'''
    assert isinstance(level, LogLevel)
    level_name = str(level)[len(LogLevel.__name__)+1:]
    prompt = colored_prompt(level_name)

    '''inline'''
    end = " " if inline else "\n"

    '''Using LogLevel Color'''
    if color == None:
        # print(colorize(prompt, bold=True), colorize(msg, color=level.value, bold=bold))
        print(prompt, colorize(msg, color=level.value, bold=bold), end=end)
    else:
        print(colorize(">>>", bold=True), colorize(msg, color=color, bold=bold), end=end)


def debug_print(
        msg: str,
        args=Any,
        level: LogLevel = LogLevel.DEBUG,
        inline=False
    ):
    debug_msg(msg, level, inline=inline)
    print(args)


def get_formatted_time():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def get_space_dim(space):
    if isinstance(space, gym.spaces.Box):
        return space.shape[0]
    elif isinstance(space, gym.spaces.Discrete):
        return space.n

    
def test_get_space_dim():
    env = gym.make("CartPole-v1")
    debug_print("action space:", args=get_space_dim(env.action_space))
    debug_print("obs space:", args=get_space_dim(env.observation_space))


if __name__ == "__main__":
    print("="*10 + " every color " + "="*10)
    for c in Color:
        print(colorize(f"{c}", color=c, bold=False))
        print(colorize(f"{c}", color=c))

    print("")
    print("="*10 + " every level " + "="*10)
    for l in LogLevel:
        level_name = str(l)[len(LogLevel.__name__)+1:]
        debug_msg(level_name, level=l)


    print("")
    print("="*10 + " other color " + "="*10)
    debug_msg("BLUE", color=Color.BLUE)
    debug_msg("CYAN", color=Color.CYAN)
    debug_msg("GREEN", color=Color.GREEN)
    debug_msg("MAGENTA", color=Color.MAGENTA)

    print("")
    print("="*10 + " inline " + "="*10)
    debug_print("hello", args="world", inline=True)

    print("")
    print("="*10 + " newline " + "="*10)
    debug_print("hello", args="world")

    print(get_formatted_time())


    test_get_space_dim()