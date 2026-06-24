'''Main definitions.'''
from argparse import ArgumentParser, Namespace
from collections import OrderedDict, ChainMap
from inspect import signature, Signature

from typing import (Any, Callable, Dict, List, NamedTuple,
                    no_type_check_decorator, Sequence, Tuple, TypeVar, Union)

from .utils import empty_args


STOP = 0b01
REQUIRED = 0b10


class NotArg:
    '''Class to signal a given function parameter is not a CLI argument.

    Notice that if any parameter is a NotArg it should have a default
    value or the subcommand must be driven by its parent comment.
    '''
    pass


class SubparserNamespace:
    '''A class that acts as a namespace for subparsers.

    It collects arguments in the `subargs` member, so that hierarchical
    subcommands receive their arguments accordingly.
    '''
    subargs: Namespace

    def __init__(self) -> None:
        super().__setattr__('subargs', Namespace())

    def __setattr__(self, k, v) -> None:
        setattr(self.subargs, k, v)  # pylint: disable=no-member


class Arg:
    '''Class for specifying details about an argument.

    This class is used as an intermediary with the internal implementation.
    Argparse `add_argument` arguments are valid as parameters to the
    constructor.
    '''
    args: Sequence[Any]

    def __init__(self, *args, optional=False, **kwargs) -> None:
        if args and args[0].startswith('-'):
            optional = True

        self.named = bool(args and isinstance(args[0], str))
        self.optional = optional
        self.args = args

        if 'choices' in kwargs:
            kwargs['choices'] = sorted(kwargs['choices'])

        self.kwargs = kwargs


class ArgSpec(NamedTuple):
    '''Specs of an argument, namely its annotation and its default value'''
    annotation: Union[Arg, str, type]
    default: Any


def add_argument(parser, arg_details: Tuple[str, ArgSpec]) -> None:
    '''Add an argument to a parser based on details from a function parameter.

    Notice that parameters named 'subcommand' are treated specially. They are
    used to hand the subcommand object to a parent command. The default
    argument is used as a way to specify if the subcommand should wait for its
    parent to drive it back into execution (`STOP`) and/or if it is required
    (`REQUIRED`).  The flags can be bitwise-or'ed together.

    Args:
        arg_details: the name and possibly annotation and default value from a
                     function parameter.
    '''
    name, spec = arg_details
    arg, default = spec

    if name == 'subcommand':
        parser.ensure_subparsers()
        if default is Signature.empty:
            parser.sub.required = True
        if isinstance(default, int):
            if default & STOP:
                parser.stop = True
            if default & REQUIRED:
                parser.sub.required = True
        return

    if arg is Signature.empty:
        arg = Arg()

    elif isinstance(arg, str):
        arg = Arg(help=arg)

    elif isinstance(arg, type):
        arg = Arg(type=arg)

    if default is not Signature.empty:
        arg.kwargs['default'] = default
        arg.optional = True

    if arg.optional:
        arg.kwargs['dest'] = name

    name = '--' + name if arg.optional else name

    if not arg.named:
        arg.args = [name] + list(arg.args)

    parser.add_argument(*(arg.args), **(arg.kwargs))


def arg_format(sig: Signature) -> Dict[str, ArgSpec]:
    '''Make the signature of a function into an ordered dict.

    Args:
        sig: a function signature from `inspect`
    '''
    return OrderedDict((a.name, ArgSpec(a.annotation, a.default))
                       for a in sig.parameters.values()
                       if a.annotation is not NotArg)


T = TypeVar('T')


class Command(ArgumentParser):
    '''A wrapper for an entry-point function capable of parsing arguments.

    Args:
        f: the function to wrap.
        is_sub: flags if the command is a subcommand (think `add` in
                `git add`).
        args: forwarded to ArgumentParser.
        kwargs: forwarded to ArgumentParser.
    '''
    format: Dict[str, ArgSpec]

    def __init__(self,
                 f: Callable[..., T]=None,
                 is_sub: bool = False,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sub: Any = None
        self.format = None
        self.is_sub = is_sub
        self.subcommands: List[Any] = []
        self.inner = f
        self.unwrapped = f
        self.stop = False

    def ensure_subparsers(self) -> None:
        '''Initialize the subparsers from ArgumentParser before adding any.'''
        if self.sub is None:
            self.sub = self.add_subparsers(dest='subcommand')

    def run(self, argv: Sequence[str] = None) -> None:
        '''Run the inner function with arguments from an argument list.

        This is the function to use to actually run your program.

        Arguments:
            argv: the argument list from command line. Usually sys.argv[1:].
        '''
        arguments = self.parse_args(argv)

        self.resume(arguments)

    async def run_async(self, argv: Sequence[str] = None) -> None:
        '''Run the inner async function with arguments from an argument list.

        This is the function to use to actually run your program.

        Arguments:
            argv: the argument list from command line. Usually sys.argv[1:].
        '''
        arguments = self.parse_args(argv)

        await self.resume_async(arguments)

    @staticmethod
    def resume(arguments, **extra) -> None:
        '''Resume a stopped subparser.

        If a command receives its subparser with STOP as a default argument, it
        will parse its arguments and wait to be resume with possible extra
        arguments by its parent command.
        '''
        current = arguments
        while current is not None:
            vargs = vars(current)
            argd = {a: vargs[a]
                    for a in current._parser.format}

            if 'subcommand' in argd and argd['subcommand'] is not None:
                name = argd['subcommand']
                argd['subcommand'] = (current.subargs._parser,
                                      current.subargs,
                                      name)

            current._parser.inner(**ChainMap(argd, extra))

            if current._parser.stop:
                current = None
            else:
                current = getattr(current, 'subargs', None)
                extra = {}

    @staticmethod
    async def resume_async(arguments, **extra) -> None:
        '''Resume an async stopped subparser.

        If a command receives its subparser with STOP as a default argument, it
        will parse its arguments and wait to be resume with possible extra
        arguments by its parent command.
        '''
        current = arguments
        while current is not None:
            vargs = vars(current)
            argd = {a: vargs[a]
                    for a in current._parser.format}

            if 'subcommand' in argd and argd['subcommand'] is not None:
                name = argd['subcommand']
                argd['subcommand'] = (current.subargs._parser,
                                      current.subargs,
                                      name)

            await current._parser.inner(**ChainMap(argd, extra))

            if current._parser.stop:
                current = None
            else:
                current = getattr(current, 'subargs', None)
                extra = {}

    def parse_known_args(self,
                         args=None,
                         namespace=None) -> Tuple[Namespace, List[str]]:
        '''Parse args normally or to a SubparserNamesace, for subcommands.'''
        if namespace is None and self.is_sub:
            namespace = SubparserNamespace()
        return super().parse_known_args(args, namespace)

    @empty_args
    @no_type_check_decorator
    def subcommand(self,
                   names: Sequence[str] = (),
                   wrapper: Callable[[Callable], Any] = None) -> Callable:
        '''Similar to the top-level `command` function, for subcommands.

        Use this to create subcommands that are subordinate to your main
        program. Think `git add` or `pip install`.

        Args:
            names: a list of alternative names to use for the subcommand.
            wrapper: a wrapper function (for example, to pretty print a
                     returned list of results.
        '''
        def inner(f: Callable):
            '''Helper function to apply `make_parser`.'''
            name, *aliases = [f.__name__] if not names else names
            self.subcommands.extend([name, *aliases])
            self.ensure_subparsers()

            def factory(*args, **kwargs) -> 'Command':
                '''Closure factory function for subparsers.'''
                return self.sub.add_parser(name, aliases=aliases,
                                           *args, **kwargs)
            parser = make_parser(factory,
                                 f, wrapper, is_sub=True)
            return parser
        return inner

    def register_subcommands(self, subcommands: Sequence[Callable]) -> None:
        '''Register a collection of subcommands manually. The `subcommand`
           decorator is usually to be preferred.'''
        for sub in subcommands:
            self.subcommand(sub)

    def __call__(self, *args, **kwargs) -> T:
        return self.unwrapped(*args, **kwargs)


def make_parser(factory: Callable[..., Command],
                function,
                wrapper=None,
                is_sub=False):
    '''Create an argument parser from a factory function and an entry-point.'''
    sig = signature(function)
    cmd_function = wrapper(function) if wrapper else function

    options = sig.return_annotation

    if isinstance(options, str):
        options = {
            'help': options,
            'description': options
        }
    elif options is Signature.empty and function.__doc__:
            description = function.__doc__
            if is_sub:
                options = {
                    'help': description,
                    'description': description
                }
            else:
                options = {'description': description}
    else:
        options = {}

    parser = factory(**options)

    parser.inner = cmd_function
    parser.unwrapped = function
    parser.format = arg_format(sig)
    parser.is_sub = is_sub

    for argspec in parser.format.items():
        add_argument(parser, argspec)

    parser.set_defaults(_parser=parser)
    return parser


@no_type_check_decorator
@empty_args
def command(wrapper=None) -> Callable:
    '''Create a wrapper for a function that serves as an entry for the program.

    The returned decorator creates a top-level command, usually a main-like
    function. Can be called with or without arguments, as a "magic decorator".
    '''
    def inner(f):
        '''Wrap an entry-point function through `make_parser`.'''
        return make_parser(Command, f, wrapper=wrapper)
    return inner
