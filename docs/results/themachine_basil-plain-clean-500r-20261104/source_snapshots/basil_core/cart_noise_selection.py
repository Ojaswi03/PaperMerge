"""Flexible noise selection around the unchanged scientific configurations."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NoiseSelection:
    count:int=1
    values:tuple[float,...]=(.01,)

    @classmethod
    def create(cls,count=1,values=None):
        defaults={1:(.01,),2:(.005,.02),3:(.005,.01,.02)}
        if type(count) is not int or count not in defaults:raise ValueError('Noise levels must be 1, 2 or 3.')
        selected=tuple(float(v) for v in (defaults[count] if values is None else values))
        if len(selected)!=count:raise ValueError(f'--noise-levels {count} requires exactly {count} noise values.')
        if len(set(selected))!=count or any(not math.isfinite(v) or v<=0 for v in selected):
            raise ValueError('Noise standard deviations must be distinct, finite and positive.')
        return cls(count,tuple(sorted(selected)))


def configurations(args):
    from basil_core.sharp_study import configuration,CONDITION_NAMES
    from basil_core.factorial_study import CONDITIONS
    from basil_core.sharp_guarded_protocol import guarded_configuration
    from basil_core.sharp_cart_v2_protocol import configuration as cart_configuration
    selection=NoiseSelection.create(args.noise_levels,args.noise_values)
    for name,values in [('datasets',args.datasets),('partitions',args.partitions),
            ('seeds',args.seeds),('conditions',args.conditions)]:
        if not values or len(set(values))!=len(values):raise ValueError(name+' must be nonempty and distinct.')
    configs=[]
    for dataset in args.datasets:
        for partition in args.partitions:
            for seed in args.seeds:
                for condition in args.conditions:
                    flags=CONDITIONS[CONDITION_NAMES.index(condition)]
                    for sigma in (selection.values if flags.noise else (0.,)):
                        base=configuration(dataset,partition,seed,condition,sigma,nodes=args.nodes,
                            rounds=args.rounds,local_steps=args.local_steps,model=args.model,alpha=args.alpha,
                            attack_start=args.attack_start,train_per_class=args.train_per_class,
                            validation_per_class=args.validation_per_class)
                        configs.append(cart_configuration(guarded_configuration(base)))
    return selection,configs
