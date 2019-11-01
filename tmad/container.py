from .generator import Generator
from .discriminator import Discriminator

import torch
import torchvision.utils
import os.path

# Outputs that the discriminator should aim for
REAL_LABEL = 1
FAKE_LABEL = 0


class Container:
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        workdir: str,
    ):
        self._dataloader = dataloader
        self._workdir = workdir

        self.__init_workdirs()
        self.__init_devices()
        self.__init_generator()
        self.__init_discriminator()

        self.hyperparameters = {
            'learning_rate': 0.0002,
            'adam_beta': (
                0.5,    # Configurable in tutorial code
                0.999,  # Hardcoded in tutorial code
            ),
        }

        self._generator_optimizer = torch.optim.Adam(
            params = self._discriminator.parameters(),
            lr     = self.hyperparameters['learning_rate'],
            betas  = self.hyperparameters['adam_beta'],
        )

        self._discriminator_optimizer = torch.optim.Adam(
            params = self._generator.parameters(),
            lr     = self.hyperparameters['learning_rate'],
            betas  = self.hyperparameters['adam_beta'],
        )

        self._criterion = torch.nn.BCELoss()

        # TODO make configurable?
        # I don't know exactly what this signifies atm
        self._latent_vector_size = 100

        self.generator_losses = []
        self.discriminator_losses = []

        self._fixed_noise = torch.randn(
            *(
                self._dataloader.batch_size,
                self._latent_vector_size,
                1, 1,
            ),
            device=self._first_device,
        )
    def __init_devices(self):
        self._ngpus = torch.cuda.device_count()
        self._devices = []

        if (self._ngpus == 0):
            raise Exception("At least one GPU required")

        for i in range(self._ngpus):
            self._devices.append(torch.device(type='cuda', index=i))

    @staticmethod
    def __weights_init(m):
        # This seems like a fucking weird-ass way to initialise the weights,
        # but for now I'm just copying the Pytorch tutorial

        # Recommended magic constants!
        magic_gaussian_stddev = 0.02

        classname = m.__class__.__name__
        if 'Conv' in classname:
            # Initialise the weights from a normal distribution (Gaussian)...
            torch.nn.init.normal_(
                tensor = m.weight.data,
                mean   = 0.0,
                std    = magic_gaussian_stddev,
            )

        elif 'BatchNorm' in classname:
            # Gaussian again for the weights, but this time with a mean of 1.0
            torch.nn.init.normal_(
                tensor = m.weight.data,
                mean   = 1.0,
                std    = magic_gaussian_stddev,
            )

            # Zero the biases
            torch.nn.init.constant_(
                tensor = m.bias.data,
                val    = 0,
            )

    def __init_workdirs(self):
        subdirs = [
            'nets/generator',
            'nets/discriminator',
            'training_fakes',
        ]

        for path in subdirs:
            os.makedirs(
                os.path.join(self._workdir, path),
                mode=0o755,
            )

    def __init_generator(self):
        g = Generator(
            feature_map_size = 64,
            input_size       = 100,
            color_channels   = 3,
        )

        g.to(self._first_device)
        if self._ngpus > 1:
            g = nn.DataParallel(g, list(range(self._ngpus)))

        g.apply(Container.__weights_init)
        
        self._generator = g

    def __init_discriminator(self):
        d = Discriminator(
            feature_map_size = 64,
            color_channels   = 3,
        )

        d.to(self._first_device)
        #if self._ngpu > 1:
        #    d = nn.DataParallel(d, list(range(self._ngpus)))

        d.apply(Container.__weights_init)

        self._discriminator = d

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for batch_number, batch_data in enumerate(self._dataloader, start=0):

                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ## Train with an all-real batch
                self._discriminator.zero_grad()

                # batch_data is a list with two items
                # the first item is a list of length batch_size containing the image
                # the second item is a list of length batch_size containing zeroes
                # I don't yet understand why that is, or what the second item is.
                real_data = batch_data[0].to(self._first_device)
                batch_size = real_data.size(0)

                label = torch.full(
                    size       = (batch_size,),
                    fill_value = REAL_LABEL,
                    device     = self._first_device,
                )

                # Forward-pass batch of real images through discriminator
                output = self._discriminator(real_data).view(-1)

                # Calculate loss on all-real batch
                errD_real = self._criterion(output, label)

                # Calculate gradients for D in backward pass
                errD_real.backward()

                D_x = output.mean().item()


                ## Train with an all-fake batch
                fakes = self._generate_fakes(batch_size)
                label.fill_(FAKE_LABEL)

                # Forward-pass batch of fake images through discriminator
                output = self._discriminator(fakes.detach()).view(-1)

                errD_fake = self._criterion(output, label)

                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake

                # Update Discriminator network
                self._discriminator_optimizer.step()


                ## (2) Update G network: maximise log(D(G(z)))
                self._generator.zero_grad()

                # Fake labels are real for generator cost
                label.fill_(REAL_LABEL)

                # Since we just updated D, run the batch of fakes through it again
                # Hmm, in the example code this is fakes not fakes.detach() as before?
                output = self._discriminator(fakes).view(-1)

                # Calculate G's loss based on this output
                errG = self._criterion(output, label)

                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()

                # Update G
                self._generator_optimizer.step()

                # Output training stats
                if batch_number % 50 == 0:
                    print(
                        f'[Epoch {epoch} / {num_epochs}] '
                        f'[Batch {batch_number} / {len(self._dataloader)}] '
                        f'[Loss D: {errD.item():.4f}] '
                        f'[Loss G: {errG.item():.4f}] '
                        f'[D(x): {D_x:.4f}] '
                        f'[D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}]'
                    )

                # Save losses
                self.generator_losses.append(errG.item())
                self.discriminator_losses.append(errD.item())

                # Check how the generator is doing by saving its output from fixed_noise
                if batch_number % 100 == 0:
                    fixed_fakes = self._generator(self._fixed_noise)
                    
                    filename = os.path.join(
                        self._workdir,
                        'training_fakes',
                        f'epoch_{epoch:03d}.png',
                    )
                    
                    torchvision.utils.save_image(
                        fixed_fakes.detach(),
                        filename,
                        normalize=True,
                    )

            # Do checkpointing (saving the state of the nets?)
            torch.save(
                self._generator.state_dict(),
                f'nets/generator/epoch_{epoch:03d}',
            )
            torch.save(
                self._discriminator.state_dict(),
                f'nets/discriminator/epoch_{epoch:03d}',
            )

    @property
    def _batch_size(self):
        return self._dataloader.batch_size

    @property
    def _first_device(self):
        return self._devices[0]

    def _generate_fakes(self, batch_size):
        noise_shape = (
            batch_size,
            self._latent_vector_size,
            1,
            1,
        )

        noise = torch.randn(
            *noise_shape,
            device=self._first_device,
        )

        return self._generator(noise)

