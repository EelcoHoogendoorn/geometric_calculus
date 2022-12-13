import jax


def optimize(model, params, objectives, n_steps=101, learning_rate=3e-3):
	import optax
	tx = optax.adam(learning_rate=learning_rate)
	opt_state = tx.init(params)

	def make_step(obj, sampler, weight):
		# handle vmapping over batches of samples
		def inner_step(params, key):
			return obj(model(params), sampler(key)) * weight
		qq = jax.vmap(inner_step, in_axes=(None, 0))    # vmap over key/samples
		return lambda p, k: qq(p, k).mean() # mean over samples

	steps = [make_step(obj, sampler, w) for obj, sampler, n, w in objectives]
	n_samples = [n for obj, sampler, n, w in objectives]

	# grads wrt params, and jit
	loss_grad_fns = [jax.jit(jax.value_and_grad(s)) for s in steps]

	key = jax.random.PRNGKey(0)
	for i in range(n_steps):
		if i % 100 == 0:
			learning_rate *= 0.9

		if i % 10 == 0:
			print('Loss step {}: '.format(i))
		for n, lg in zip(n_samples, loss_grad_fns):
			key, _ = jax.random.split(key)
			loss_val, grads = lg(params, jax.random.split(key, n))
			updates, opt_state = tx.update(grads, opt_state)
			if i % 10 == 0:
				print(loss_val)

		params = optax.apply_updates(params, updates)
	return params
