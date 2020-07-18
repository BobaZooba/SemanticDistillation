# class BaseConfig(Namespace):
#
#     def __init__(self,
#                  data_dir: str = 'data/convai2/',
#                  model_type: str = 'bert-base-uncased',
#                  data_type: str = 'basic',
#                  max_length: int = 32,
#                  max_candidates: int = 20,
#                  question_token_type_id: int = 1,
#                  response_token_type_id: int = 2,
#                  batch_size: int = 256,
#                  candidates_batch_size: int = 8,
#                  tokenize_batch_size: int = 2048,
#                  verbose: bool = True):
#         super().__init__(data_dir=data_dir,
#                          model_type=model_type,
#                          data_type=data_type,
#                          max_length=max_length,
#                          max_candidates=max_candidates,
#                          question_token_type_id=question_token_type_id,
#                          response_token_type_id=response_token_type_id,
#                          batch_size=batch_size,
#                          candidates_batch_size=candidates_batch_size,
#                          tokenize_batch_size=tokenize_batch_size,
#                          verbose=verbose)
#
#
# class DatasetPreparer:
#
#     TRAIN_FILE = 'train.json'
#     TRAIN_WITH_CANDIDATES_FILE = 'train_with_candidates.json'
#     VALID_FILE = 'valid.json'
#     VALID_WITH_CANDIDATES_FILE = 'valid_with_candidates.json'
#
#     INDEX_TO_TEXT_FILE = 'index_to_text.json'
#
#     def __init__(self, config: Namespace):
#
#         self.config = config
#
#         self.data_dir = os.path.join(os.getcwd(), self.config.data_dir)
#         self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_type)
#
#         setattr(self.config, 'pad_index', self.tokenizer.pad_token_id)
#
#         self.index_to_text = self.build_index()
#
#     def load_json_file(self, file: str):
#
#         with open(os.path.join(self.data_dir, file)) as file_object:
#             data = json.load(file_object)
#
#         return data
#
#     def build_index(self):
#
#         index_to_text = dict()
#
#         for key, value in self.load_json_file(self.INDEX_TO_TEXT_FILE).items():
#             index_to_text[int(key)] = value
#
#         if self.config.data_type in ('text',):
#             return index_to_text
#
#         index_to_tokenized_text = list()
#
#         indices = list(range(len(index_to_text)))
#
#         for i_batch in tqdm(range(math.ceil(len(index_to_text) / self.config.tokenize_batch_size)),
#                             desc='Building index',
#                             disable=not self.config.verbose):
#
#             start = i_batch * self.config.tokenize_batch_size
#             stop = (i_batch + 1) * self.config.tokenize_batch_size
#
#             batch = [index_to_text[i] for i in indices[start:stop]]
#
#             tokenized_batch = self.tokenizer.batch_encode_plus(batch,
#                                                                truncation=True,
#                                                                max_length=self.config.max_length)['input_ids']
#
#             index_to_tokenized_text.extend(tokenized_batch)
#
#         return index_to_tokenized_text
#
#     def build_data(self, file, with_candidates: bool):
#
#         if self.config.data_type in ('basic', 'common'):
#             dataset = ConvAI2Dataset(data=self.load_json_file(file=file),
#                                      index_to_text=self.index_to_text,
#                                      with_candidates=with_candidates,
#                                      max_candidates=self.config.max_candidates,
#                                      question_token_type_id=self.config.question_token_type_id,
#                                      response_token_type_id=self.config.response_token_type_id,
#                                      max_length=self.config.max_length,
#                                      pad_index=self.config.pad_index)
#         elif self.config.data_type in ('text',):
#             dataset = ConvAI2TextDataset(data=self.load_json_file(file=file),
#                                          index_to_text=self.index_to_text,
#                                          with_candidates=with_candidates,
#                                          max_candidates=self.config.max_candidates)
#         else:
#             raise ValueError('Not available data_type')
#
#         return dataset
#
#     def load_data(self, as_data_loader: bool = False):
#
#         train_data = self.build_data(file=self.TRAIN_FILE, with_candidates=False)
#         train_with_candidates_data = self.build_data(file=self.TRAIN_WITH_CANDIDATES_FILE, with_candidates=True)
#
#         valid_data = self.build_data(file=self.VALID_FILE, with_candidates=False)
#         valid_with_candidates_data = self.build_data(file=self.VALID_WITH_CANDIDATES_FILE, with_candidates=True)
#
#         if as_data_loader:
#             train_loader = DataLoader(dataset=train_data,
#                                       batch_size=self.config.batch_size,
#                                       shuffle=True,
#                                       collate_fn=train_data.collate,
#                                       drop_last=True)
#
#             valid_loader = DataLoader(dataset=valid_data,
#                                       batch_size=self.config.batch_size,
#                                       collate_fn=valid_data.collate,
#                                       drop_last=True)
#
#             train_with_candidates_loader = DataLoader(dataset=train_with_candidates_data,
#                                                       batch_size=self.config.candidates_batch_size,
#                                                       shuffle=True,
#                                                       collate_fn=train_with_candidates_data.collate)
#
#             valid_with_candidates_loader = DataLoader(dataset=valid_with_candidates_data,
#                                                       batch_size=self.config.candidates_batch_size,
#                                                       collate_fn=valid_with_candidates_data.collate)
#
#             data = (train_loader, valid_loader)
#             data_with_candidates = (train_with_candidates_loader, valid_with_candidates_loader)
#
#         else:
#             data = (train_data, valid_data)
#             data_with_candidates = (train_with_candidates_data, valid_with_candidates_data)
#
#         return data, data_with_candidates
#
#
# class Recall:
#
#     def __init__(self, k_variants: Tuple[int] = (1, 3, 5), c_variants: Tuple[int] = (2, 5, 10, 15, 20)):
#
#         self.k_variants = k_variants
#         self.c_variants = c_variants
#
#         self.matrices = None
#         self._messages = list()
#         self.step = 0
#
#         self.reset()
#
#     def add(self, similarity_matrix: np.array):
#         self.matrices.append(similarity_matrix)
#
#     def reset(self):
#         self.matrices = list()
#
#     def calculate(self, k: int, c: int):
#         similarity_matrix = torch.cat(self.matrices)[:, :c]
#
#         ranked = similarity_matrix.argsort(descending=True)
#         ranked = ranked[:, :k] == 0
#
#         ranked = ranked.sum(dim=-1).float()
#
#         recall = (ranked.sum() / ranked.shape[0]).item()
#
#         return recall
#
#     @property
#     def metrics(self):
#
#         metrics = list()
#         self.step += 1
#
#         if len(self._messages) > 0:
#             self._messages.append(30 * '=')
#
#         for k in self.k_variants:
#
#             metrics.append(list())
#
#             for c in self.c_variants:
#
#                 if k >= c:
#                     metrics[-1].append(np.NaN)
#                 else:
#                     current_metric = round(self.calculate(k, c), 3)
#                     self._messages.append(f'Step {self.step} | Recall @ {k}/{c}: {current_metric:.3f}')
#                     metrics[-1].append(current_metric)
#
#         metrics = pd.DataFrame(data=metrics)
#
#         metrics.index = [f'@ {i}' for i in self.k_variants]
#         metrics.columns = [f'n_candidates {i}' for i in self.c_variants]
#
#         return metrics
#
#     @property
#     def messages(self):
#         return '\n'.join(self._messages)
#
#
#
# def train_loop(query_encoder, response_encoder, loader, criterion, optimizer, clip_grad_norm=3.):
#     losses = list()
#     progress_bar = tqdm(total=len(loader), desc='Training')
#
#     query_encoder.train()
#     response_encoder.train()
#
#     for token_ids, positions, token_types in loader:
#
#         query_token_ids = token_ids[0].to(device)
#         response_token_ids = token_ids[1].to(device)
#
#         optimizer.zero_grad()
#
#         query_embed = query_encoder(query_token_ids)
#         response_embed = response_encoder(response_token_ids)
#
#         loss = criterion(query_embed, response_embed)
#         loss.backward()
#
#         if clip_grad_norm > 0:
#             for param_group in optimizer.param_groups:
#                 torch.nn.utils.clip_grad_norm_(param_group['params'], clip_grad_norm)
#
#         optimizer.step()
#
#         progress_bar.update()
#         losses.append(loss.item())
#
#         progress_bar.set_postfix(loss=np.mean(losses[-100:]))
#
#     progress_bar.close()
#
#     return losses
#
#
# def evaluation_loop(recall, query_encoder, response_encoder, loader):
#
#     recall.reset()
#
#     query_encoder.eval()
#     response_encoder.eval()
#
#     for token_ids, positions, token_types in tqdm(loader, desc='Evaluation'):
#
#         query_token_ids = token_ids[0].to(device)
#         response_token_ids = token_ids[1].to(device)
#
#         with torch.no_grad():
#             query_embed = query_encoder(query_token_ids)
#             response_embed = response_encoder(response_token_ids)
#
#             similarity_matrix = score_candidates(query_embed, response_embed)
#
#         recall.add(similarity_matrix)
#
#
# def score_candidates(question_embeddings, candidates_embeddings):
#     candidates_batch_size, model_dim = candidates_embeddings.size()
#     candidates_per_sample = candidates_batch_size // question_embeddings.size(0)
#
#     candidates_embeddings = candidates_embeddings.view(question_embeddings.size(0),
#                                                        candidates_per_sample,
#                                                        model_dim)
#
#     similarity_matrix = torch.bmm(question_embeddings.unsqueeze(1),
#                                   candidates_embeddings.transpose(1, 2)).squeeze(dim=1)
#
#     similarity_matrix = similarity_matrix.detach().cpu()
#
#     return similarity_matrix
#
#
# epochs = 10
#
# valid_recall = data.Recall()
#
# for n_epoch in range(epochs):
#     print(f'Epoch {n_epoch + 1}')
#
#     train_losses = train_loop(query_encoder, response_encoder, train_loader, criterion, optimizer)
#     valid_recall.reset()
#     evaluation_loop(valid_recall, query_encoder, response_encoder, valid_loader_with_candidates)
#     display(valid_recall.metrics)
