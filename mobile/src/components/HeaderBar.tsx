import { Feather } from "@expo/vector-icons";
import React from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";

import { colors, fonts } from "../theme";

interface HeaderBarProps {
  onToggleSidebar: () => void;
  onNewChat: () => void;
}

export function HeaderBar({ onToggleSidebar, onNewChat }: HeaderBarProps) {
  return (
    <View style={styles.container}>
      <TouchableOpacity
        onPress={onToggleSidebar}
        style={styles.iconButton}
        accessibilityLabel="Toggle sidebar"
      >
        <Feather name="menu" size={20} color={colors.foreground} />
      </TouchableOpacity>
      <View style={styles.brandPill}>
        <Text style={styles.brandEmoji}>LAW</Text>
        <Text style={styles.brandText}>The Law Says</Text>
      </View>
      <TouchableOpacity
        onPress={onNewChat}
        style={styles.iconButton}
        accessibilityLabel="Start new chat"
      >
        <Feather name="plus" size={20} color={colors.foreground} />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    height: 60,
    paddingHorizontal: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    backgroundColor: "rgba(5,5,5,0.95)"
  },
  iconButton: {
    height: 40,
    width: 40,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    alignItems: "center",
    justifyContent: "center"
  },
  brandPill: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card
  },
  brandEmoji: {
    color: colors.foreground,
    fontSize: 14,
    marginRight: 6
  },
  brandText: {
    color: colors.foreground,
    fontFamily: fonts.medium,
    fontSize: 14
  }
});
