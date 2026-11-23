<script setup>
import { PhShuffle } from '@phosphor-icons/vue'
import { t } from '../../i18n'
const seed = defineModel('seed', { default: '' })
defineProps({ ready: Boolean, submitting: Boolean, canSubmit: Boolean, error: String, label: String, seedVisible: { type: Boolean, default: true } })
</script>
<template>
  <div v-if="seedVisible" class="field">
    <label for="seed" class="field-label">{{ t('Seed') }}<span class="aside">{{ t('Empty for random') }}</span></label>
    <div class="seed">
      <input id="seed" v-model="seed" class="input mono" type="text" inputmode="numeric" :placeholder="t('Random')" />
      <button type="button" class="btn icon" :aria-label="t('Pick a random seed')" @click="seed = String(Math.floor(Math.random() * 2 ** 40))"><PhShuffle :size="16" /></button>
    </div>
  </div>
  <div class="submit">
    <span class="estimate"><slot /></span>
    <button class="btn primary go" :disabled="!canSubmit">{{ submitting ? t('Submitting') : label }}</button>
  </div>
  <p v-if="!ready" class="help">{{ t('The server is still loading models; you can submit once it is ready.') }}</p>
  <p v-if="error" class="error" role="alert">{{ error }}</p>
</template>
