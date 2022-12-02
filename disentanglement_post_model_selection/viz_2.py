
                # post_mean = torch.zeros_like(post_logvar) + 0.2 # Jun 14 2022
                # print("post_mean.shape") # Jun 14 2022
                # print(post_mean.shape) # Jun 14 2022
                # print("post_mean") # Jun 14 2022
                # print(post_mean) # Jun 14 2022
                # print("post_logvar") # Jun 14 2022
                # print(post_logvar) # Jun 14 2022
                samples = self.model.reparameterize(post_mean, post_logvar)
                # print("samples") # Jun 14 2022
                # print(samples) # Jun 14 2022
                samples = samples.cpu().repeat(n_samples, 1)
                # print("samples") # Jun 14 2022
                # print(samples) # Jun 14 2022
                post_mean_idx = post_mean.cpu()[0, idx]
                # print("post_mean_idx") # Jun 14 2022
                # print(post_mean_idx) # Jun 14 2022
                post_std_idx = torch.exp(post_logvar / 2).cpu()[0, idx]
                # print("post_std_idx") # Jun 14 2022
                # print(post_std_idx) # Jun 14 2022

            # travers from the gaussian of the posterior in case quantile
            traversals = torch.linspace(*self._get_traversal_range(mean=post_mean_idx,
                                                                   std=post_std_idx),
                                        steps=n_samples)

        # print("traversals") # Jun 14 2022
        # print(traversals) # Jun 14 2022

        for i in range(n_samples):
            samples[i, idx] = traversals[i]

        # print("samples") # Jun 14 2022
        # print(samples) # Jun 14 2022

        return samples

    def _save_or_return(self, to_plot, size, filename, is_force_return=False):
        """Create plot and save or return it."""
        to_plot = F.interpolate(to_plot, scale_factor=self.upsample_factor)

        if size[0] * size[1] != to_plot.shape[0]:
            raise ValueError("Wrong size {} for datashape {}".format(size, to_plot.shape))

        # `nrow` is number of images PER row => number of col
        kwargs = dict(nrow=size[1], pad_value=(1 - get_background(self.dataset)))
        if self.save_images and not is_force_return:
            filename = os.path.join(self.model_dir, self.experiment_name + '_' + filename)
            save_image(to_plot, filename, **kwargs)
        else:
            return make_grid_img(to_plot, **kwargs)

    def _decode_latents(self, latent_samples):
        """Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        # print("latent_samples.shape") # Jun 14 2022
        # print(latent_samples.shape) # Jun 14 2022
        # print("latent_samples") # Jun 14 2022
        # print(latent_samples) # Jun 14 2022
        latent_samples = latent_samples.to(self.device)
        return self.model.decoder(latent_samples).cpu()

    def generate_samples(self, size=(8, 8)):
        """Plot generated samples from the prior and decoding.

        Parameters
        ----------
        size : tuple of ints, optional
            Size of the final grid.
        """
        prior_samples = torch.randn(size[0] * size[1], self.latent_dim)
        generated = self._decode_latents(prior_samples)
        return self._save_or_return(generated.data, size, PLOT_NAMES["generate_samples"])

    def data_samples(self, data, size=(6, 6)):
        """Plot samples from the dataset

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints, optional
            Size of the final grid.
        """
        data = data[:size[0] * size[1], ...]
        return self._save_or_return(data, size, PLOT_NAMES["data_samples"])

    def reconstruct(self, data, size=(8, 8), is_original=True, is_force_return=False):
#    def reconstruct(self, data, location, brand, circa, movement, diameter, material, timetrend, filenames, size=(8, 8), is_original=True, is_force_return=False):
        """Generate reconstructions of data through the model.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints, optional
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even when `is_original`, so that upper
            half contains true data and bottom half contains reconstructions.contains

        is_original : bool, optional
            Whether to exclude the original plots.

        is_force_return : bool, optional
            Force returning instead of saving the image.
        """
        if is_original:
            if size[0] % 2 != 0:
                raise ValueError("Should be even number of rows when showing originals not {}".format(size[0]))
            n_samples = size[0] // 2 * size[1]
        else:
            n_samples = size[0] * size[1]

        with torch.no_grad():
            originals = data.to(self.device)[:n_samples, ...]
            location = location.to(self.device)[:n_samples, ...]
            brand = brand.to(self.device)[:n_samples, ...]
            circa = circa.to(self.device)[:n_samples, ...]
            movement = movement.to(self.device)[:n_samples, ...]
            diameter = diameter.to(self.device)[:n_samples, ...]
            material = material.to(self.device)[:n_samples, ...]
            timetrend = timetrend.to(self.device)[:n_samples, ...]
            filenames = filenames.to(self.device)[:n_samples, ...]
            recs, _, _, _, _ = self.model(originals,location,brand,circa,movement,diameter,material,timetrend,filenames)

        originals = originals.cpu()
        recs = recs.view(-1, *self.model.img_size).cpu()

        to_plot = torch.cat([originals, recs]) if is_original else recs
        return self._save_or_return(to_plot, size, PLOT_NAMES["reconstruct"],
                                    is_force_return=is_force_return)

    def traversals(self,
                   data=None,
                   is_reorder_latents=False,
                   n_per_latent=8,
                   n_latents=None,
                   is_force_return=False):
        """Plot traverse through all latent dimensions (prior or posterior) one
        by one and plots a grid of images where each row corresponds to a latent
        traversal of one latent dimension.

        Parameters
        ----------
        data : bool, optional
            Data to use for computing the latent posterior. If `None` traverses
            the prior.

        n_per_latent : int, optional
            The number of points to include in the traversal of a latent dimension.
            I.e. number of columns.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        is_reorder_latents : bool, optional
            If the latent dimensions should be reordered or not

        is_force_return : bool, optional
            Force returning instead of saving the image.
        """
        n_latents = n_latents if n_latents is not None else self.model.latent_dim
        latent_samples = [self._traverse_line(dim, n_per_latent, data=data)
                          for dim in range(self.latent_dim)]
        decoded_traversal = self._decode_latents(torch.cat(latent_samples, dim=0))
        # print("decoded_traversal.shape") # Jun 14 2022
        # print(decoded_traversal.shape) # Jun 14 2022
        # print("decoded_traversal") # Jun 14 2022
        # print(decoded_traversal) # Jun 14 2022

        if is_reorder_latents:
            n_images, *other_shape = decoded_traversal.size()
            n_rows = n_images // n_per_latent
            decoded_traversal = decoded_traversal.reshape(n_rows, n_per_latent, *other_shape)
            decoded_traversal = sort_list_by_other(decoded_traversal, self.losses)
            decoded_traversal = torch.stack(decoded_traversal, dim=0)
            decoded_traversal = decoded_traversal.reshape(n_images, *other_shape)

        decoded_traversal = decoded_traversal[range(n_per_latent * n_latents), ...]
        # print("decoded_traversal.shape") # Jun 14 2022
        # print(decoded_traversal.shape) # Jun 14 2022
        # print("decoded_traversal") # Jun 14 2022
        # print(decoded_traversal) # Jun 14 2022

        # print("decoded_traversal.data") # Jun 14 2022
        # print(decoded_traversal.data) # Jun 14 2022


        size = (n_latents, n_per_latent)

        # print("size") # Jun 14 2022
        # print(size) # Jun 14 2022

        sampling_type = "prior" if data is None else "posterior"
        filename = "{}_{}".format(sampling_type, PLOT_NAMES["traversals"])

        return self._save_or_return(decoded_traversal.data, size, filename,
                                    is_force_return=is_force_return)

    def reconstruct_traverse(self, data,
#    def reconstruct_traverse(self, data,location, brand,circa,movement,diameter,material,timetrend,filenames,
                             is_posterior=True,
                             n_per_latent=8,
                             n_latents=None,
                             is_show_text=False):
        """
        Creates a figure whith first row for original images, second are
        reconstructions, rest are traversals (prior or posterior) of the latent
        dimensions.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        n_per_latent : int, optional
            The number of points to include in the traversal of a latent dimension.
            I.e. number of columns.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        is_posterior : bool, optional
            Whether to sample from the posterior.

        is_show_text : bool, optional
            Whether the KL values next to the traversal rows.
        """
        n_latents = n_latents if n_latents is not None else self.model.latent_dim

        for i in range(10):
            traversals = self.traversals(data=data[i:i+1, ...] if is_posterior else None,is_reorder_latents=True,n_per_latent=n_per_latent,n_latents=n_latents,is_force_return=True)
            traversals = Image.fromarray(traversals)
            if is_show_text:
                losses = sorted(self.losses, reverse=True)[:n_latents]
                labels = ["KL={:.4f}".format(l) for l in losses]
                traversals = add_labels(traversals, labels)
       
                filename = os.path.join(self.model_dir, self.experiment_name + '_' + str(i) + '_' + PLOT_NAMES["reconstruct_traverse"])
                traversals.save(filename)

    def gif_traversals(self, data, n_latents=None, n_per_gif=15):
        """Generates a grid of gifs of latent posterior traversals where the rows
        are the latent dimensions and the columns are random images.

        Parameters
        ----------
        data : bool
            Data to use for computing the latent posteriors. The number of datapoint
            (batchsize) will determine the number of columns of the grid.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        n_per_gif : int, optional
            Number of images per gif (number of traversals)
        """
        n_images, _, _, width_col = data.shape
        width_col = int(width_col * self.upsample_factor)
        all_cols = [[] for c in range(n_per_gif)]
        for i in range(n_images):
            grid = self.traversals(data=data[i:i + 1, ...], is_reorder_latents=True,
                                   n_per_latent=n_per_gif, n_latents=n_latents,
                                   is_force_return=True)

            height, width, c = grid.shape
            padding_width = (width - width_col * n_per_gif) // (n_per_gif + 1)
            # split the grids into a list of column images (and removes padding)
            for j in range(n_per_gif):
                all_cols[j].append(grid[:, [(j + 1) * padding_width + j * width_col + i
                                            for i in range(width_col)], :])

        pad_values = (1 - get_background(self.dataset)) * 255
        all_cols = [concatenate_pad(cols, pad_size=2, pad_values=pad_values, axis=1)
                    for cols in all_cols]

        filename = os.path.join(self.model_dir, self.experiment_name + '_' + PLOT_NAMES["gif_traversals"])
        imageio.mimsave(filename, all_cols, fps=FPS_GIF)

    def save_cbc_images(self):
        print("I am here 1") # Jun 15 2022
        torch.cuda.empty_cache()

