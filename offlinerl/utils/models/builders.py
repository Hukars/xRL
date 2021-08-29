from typing import List, Sequence, cast

import torch
import torch.nn as nn

from .encoders import EncoderFactory
from .q_functions import QFunctionFactory
from .torch import (
    CategoricalPolicy,
    ConditionalVAE,
    DeterministicPolicy,
    DeterministicRegressor,
    DeterministicResidualPolicy,
    DiscreteImitator,
    EnsembleContinuousQFunction,
    EnsembleDiscreteQFunction,
    Parameter,
    DynamicsModel,
    EnsembleDynamicsModel,
    EnsembleTransition,
    EnsembleWeight,
    EnsemblePolicy,
    ClassificationTransformer,
    ClassificationLSTM,
    StableTransformerXL,
    Policy,
    ProbablisticRegressor,
    SquashedNormalPolicy,
    ValueFunction,
)


def create_discrete_q_function(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    n_ensembles: int = 1,
) -> EnsembleDiscreteQFunction:
    if q_func_factory.share_encoder:
        encoder = encoder_factory.create(observation_shape)
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            encoder = encoder_factory.create(observation_shape)
        q_funcs.append(q_func_factory.create_discrete(encoder, action_size))
    return EnsembleDiscreteQFunction(
        q_funcs, bootstrap=q_func_factory.bootstrap
    )


def create_continuous_q_function(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    n_ensembles: int = 1,
) -> EnsembleContinuousQFunction:
    if q_func_factory.share_encoder:
        encoder = encoder_factory.create_with_action(
            observation_shape, action_size
        )
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            encoder = encoder_factory.create_with_action(
                observation_shape, action_size
            )
        q_funcs.append(q_func_factory.create_continuous(encoder))
    return EnsembleContinuousQFunction(
        q_funcs, bootstrap=q_func_factory.bootstrap
    )


def create_deterministic_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> DeterministicPolicy:
    encoder = encoder_factory.create(observation_shape)
    return DeterministicPolicy(encoder, action_size)


def create_deterministic_residual_policy(
    observation_shape: Sequence[int],
    action_size: int,
    scale: float,
    encoder_factory: EncoderFactory,
) -> DeterministicResidualPolicy:
    encoder = encoder_factory.create_with_action(observation_shape, action_size)
    return DeterministicResidualPolicy(encoder, scale)


def create_squashed_normal_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
    use_std_parameter: bool = False,
) -> SquashedNormalPolicy:
    encoder = encoder_factory.create(observation_shape)
    return SquashedNormalPolicy(
        encoder,
        action_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        use_std_parameter=use_std_parameter,
    )


def create_categorical_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> CategoricalPolicy:
    encoder = encoder_factory.create(observation_shape)
    return CategoricalPolicy(encoder, action_size)


def create_conditional_vae(
    observation_shape: Sequence[int],
    action_size: int,
    latent_size: int,
    beta: float,
    encoder_factory: EncoderFactory,
) -> ConditionalVAE:
    encoder_encoder = encoder_factory.create_with_action(
        observation_shape, action_size
    )
    decoder_encoder = encoder_factory.create_with_action(
        observation_shape, latent_size
    )
    return ConditionalVAE(encoder_encoder, decoder_encoder, beta)


def create_discrete_imitator(
    observation_shape: Sequence[int],
    action_size: int,
    beta: float,
    encoder_factory: EncoderFactory,
) -> DiscreteImitator:
    encoder = encoder_factory.create(observation_shape)
    return DiscreteImitator(encoder, action_size, beta)


def create_deterministic_regressor(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> DeterministicRegressor:
    encoder = encoder_factory.create(observation_shape)
    return DeterministicRegressor(encoder, action_size)


def create_probablistic_regressor(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> ProbablisticRegressor:
    encoder = encoder_factory.create(observation_shape)
    return ProbablisticRegressor(encoder, action_size)


def create_value_function(
    observation_shape: Sequence[int], encoder_factory: EncoderFactory
) -> ValueFunction:
    encoder = encoder_factory.create(observation_shape)
    return ValueFunction(encoder)


def create_dynamics_model(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    discrete_action: bool = False,
    deterministic: bool = False,
) -> DynamicsModel:
    encoder = encoder_factory.create_with_action(
        observation_shape=observation_shape,
        action_size=action_size,
        discrete_action=discrete_action,
    )
    model = DynamicsModel(encoder, deterministic=deterministic)
    return model


def create_ensemble_transition_model(
    observation_shape: Sequence[int],
    action_size: int,
    hidden_layer_size: int=256,
    transition_layers: int=4,
    transition_init_num: int=7,
) -> EnsembleTransition:
    return EnsembleTransition(observation_shape[0], action_size, hidden_layer_size, 
                              transition_layers, transition_init_num) 


def create_ensemble_weight_model(
    observation_shape: Sequence[int],
    action_size: int,
    weight_dim: int, 
    hidden_layer_size: int=256,
    weight_layers: int=3,
    weight_init_num: int=1,
) -> EnsembleWeight:
    return EnsembleWeight(observation_shape[0], action_size, weight_dim, 
                          hidden_layer_size, weight_layers, weight_init_num) 


def create_ensemble_dynamics_model(
    model_list: List[DynamicsModel],
    transformer_encoder: ClassificationTransformer,
    observation_shape: Sequence[int],
    action_size: int,
    encoder_size: int,
    hidden_layer_size: int = 256,
) -> EnsembleDynamicsModel:
    return EnsembleDynamicsModel(model_list, transformer_encoder, observation_shape[0],
                                 action_size, hidden_layer_size, encoder_size)


def create_ensemble_policy_model(
    policy_model_list: List[Policy],
    transformer_encoder: ClassificationTransformer,
    observation_shape: Sequence[int],
    action_size: int,
    encoder_size: int,
    policy_type: str = 'vector',
    hidden_layer_size: int = 256,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
) -> EnsemblePolicy:
    """
    param policy_type:
        'obs': input is only the observation
        'transition': input is the a piece of transition
        'vector': input is the observation concatenated with a latent vector 
    """
    if policy_type == 'obs': 
        observation_size = observation_shape[0] 
        without_latent_vector = True
    elif policy_type == 'transition':
        observation_size = observation_shape[0] * 2 + action_size + 1
        without_latent_vector = True
    elif policy_type == 'vector':
        observation_size = observation_shape[0]
        without_latent_vector = False
    return EnsemblePolicy(policy_model_list, transformer_encoder, observation_size, 
                          action_size, hidden_layer_size, encoder_size, min_logstd,
                          max_logstd, without_latent_vector)


def create_transformer_classification_model(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_size: int,
    transition_num: int,
    train_task_num: int,
    device: torch.device,
) -> ClassificationTransformer:
    transition_size = observation_shape[0] * 2 + action_size + 1
    return ClassificationTransformer(embedding_size=transition_size, last_hidden=encoder_size,
                                     pad_size=transition_num, num_classes=train_task_num, 
                                     device=device, activation=torch.nn.ReLU(), dim_model=transition_size)


def create_lstm_classification_model(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_size: int,
    train_task_num: int,
    device: torch.device,
    hidden_size: int=512,
) -> ClassificationLSTM:
    transition_size = observation_shape[0] * 2 + action_size + 1
    return ClassificationLSTM(transition_size, hidden_size, encoder_size, train_task_num, device)


def create_transformer_for_rl(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_size: int,
    transition_num: int,
    device: torch.device,
    ) -> StableTransformerXL:
    transition_size = observation_shape[0] * 2 + action_size + 1
    return StableTransformerXL(d_input=transition_size, n_layers=1, n_heads=1, 
        transition_num=transition_num, encoder_size=encoder_size, device=device)

def create_parameter(shape: Sequence[int], initial_value: float) -> Parameter:
    data = torch.full(shape, initial_value, dtype=torch.float32)
    return Parameter(data)
